from drogo.function_approximation.eigenlearner_FA.eigenlearner import EigenLearner
from jax import jit, lax
from flint import arb_mat, ctx
ctx.dps = 100   # important
import numpy as np
import jax.numpy as jnp
from functools import partial
from flax import nnx
from jax import random
import wandb


class NATGRAD_Learner(EigenLearner):

    def __init__(self, env, dataset, test_set, args):
        super().__init__(env, dataset, test_set, args)

    def compute_matrix(self):
        DR = np.linalg.inv(self.R - self.P)
        # symmetrize
        DR = (DR + DR.T) / 2
        self.matrix = DR

    def compute_top_eigvec(self):
        """
        Compute top log eigenvector of DR
        """

        matrix = arb_mat(self.matrix.tolist())
        lamb, e = matrix.eig(right=True, algorithm="approx")
        lamb = np.array(lamb).astype(np.clongdouble).real.flatten()
        e = np.array(e.tolist()).astype(np.clongdouble).real.astype(np.float32)

        idx = np.flip(lamb.argsort())
        lamb = lamb[idx]
        e = e.T[idx]
        e0 = e[0]

        if e0[0] < 0:
            e0 *= -1
        # print(e0)
        assert (e0 > 0).all()

        log_e0 = np.log(e0)
        self.true_eigvec = log_e0


    def eigvec(self):
        """ 
        Return the log of eigenvector.
        """
        eigvec = compute_eigvec(self.encoder, self.test_set, self.env.terminal_idx[0]).squeeze()
        
        col_sums = jnp.sum(eigvec, axis=0)           # shape (d,)
        flip_mask = jnp.where(col_sums > 0, -1.0, 1.0) 
        eigvec *= flip_mask

        return eigvec
    

    def cos_sim(self, eigvec):
        return compute_cos_sim(eigvec, self.true_eigvec)
    

    def learn(self):
        
        eigvec = self.eigvec()
        self.norms.append((jnp.exp(eigvec) ** 2).sum(0).mean())
        self.cos_sims.append(self.cos_sim(eigvec))

        wandb.log({
            "train/cosine_similarity": self.cos_sims[-1], 
            "train/norm": self.norms[-1],
            "train/epoch": 0 
        })

        key = random.key(self.args.seed)

        update_count = 0

        for e in range(self.args.n_epochs):

            key, subkey = random.split(key)
            batches = get_batches(self.args.batch_size, subkey, *self.processed_dataset)

            key, subkey = random.split(key)
            batches_2 = get_batches(self.args.batch_size, subkey, self.processed_dataset[0])

            # loop over minibatches
            for (s, a, r, ns, nr, t), s2 in zip(batches, batches_2):
                
                s2 = s2[0]  # unwrap tupple

                step(self.encoder, self.optimizer, s, r, ns, nr, t, s2)
                update_count += 1

            eigvec = self.eigvec()
            self.norms.append((jnp.exp(eigvec) ** 2).sum(0).mean())
            self.cos_sims.append(self.cos_sim(eigvec))

            print(self.cos_sims[-1],  self.encoder.duals[0][0])

            wandb.log({
                "train/cosine_similarity": self.cos_sims[-1], 
                "train/norm": self.norms[-1],
                "train/epoch": e + 1 
            })
    



@partial(nnx.jit, static_argnums=())
def compute_cos_sim(eigvec, true_eigvec):
    eigvec -= eigvec.max()
    true_eigvec -= true_eigvec.max()
    eigvec_norm = jnp.linalg.norm(eigvec, axis=0)
    true_eigvec_norm = jnp.linalg.norm(true_eigvec, axis=0)
    return jnp.abs((eigvec * true_eigvec).sum(0)) / (eigvec_norm * true_eigvec_norm + 1e-8) # 1e-8 for numerical stability

    
@partial(nnx.jit, static_argnums=())
def compute_eigvec(encoder, test_set, terminal_idx):
    eigvec = lax.stop_gradient(encoder(test_set))
    return eigvec
    
@partial(nnx.jit, static_argnums=())
def step(encoder, optimizer, obs, rewards, next_obs, next_rewards, terminals, obs_2):

    def dr_loss_fn(encoder, obs, rewards, next_obs, terminals, obs_2):

        rewards = jnp.expand_dims(rewards, 1)
        terminals = jnp.expand_dims(terminals, 1)

        log_phi = encoder(obs)
        log_next_phi = encoder(next_obs) # !!! No anchor
        log_phi_2 = encoder(obs_2)

        barrier_coefficient = encoder.barrier_coefs

        dr_loss =  jnp.exp(-rewards) - jnp.exp(log_next_phi - log_phi) + barrier_coefficient * (jnp.exp(2 * log_phi_2).mean() - 1)

        dr_loss = lax.stop_gradient(dr_loss) * log_phi
        dr_loss = dr_loss.mean() 

        return dr_loss
    
    loss, grads = nnx.value_and_grad(dr_loss_fn)(encoder, obs, rewards, next_obs, terminals, obs_2)
    optimizer.update(grads)
    return loss

def get_batches(batch_size, key, *datasets):

    N = datasets[0].shape[0]
    idx = random.permutation(key, N)

    for start in range(0, N, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield tuple(ds[batch_idx] for ds in datasets)