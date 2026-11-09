import optax
from flax import nnx
import jax.numpy as jnp

class DR_Encoder(nnx.Module):
    """
    """
    def __init__(self, obs_dim: int, feat_dim: int, eig_dim: int, duals_initial_val: float, barrier_initial_val: float, 
                obs_type: str, rngs: nnx.Rngs, cnn_out_dim=16): 

        self.obs_type = obs_type
        self.barrier_coefs = barrier_initial_val
        self.eig_dim = eig_dim

        if obs_type == 'image':
            
            obs_dim = feat_dim

            self.conv1 = nnx.Conv(3, 16, (3,3), rngs=rngs)
            self.conv2 = nnx.Conv(16, 32, (3,3), rngs=rngs)
            self.conv3 = nnx.Conv(32, 64, (3,3), rngs=rngs)
             
            self.image_linear = nnx.Sequential(*[
                nnx.Linear(cnn_out_dim, feat_dim, rngs=rngs),
                nnx.relu,
                nnx.Linear(feat_dim, eig_dim, rngs=rngs)
            ])

        self.eig_linear = nnx.Sequential(*[
            nnx.Linear(obs_dim, feat_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(feat_dim, feat_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(feat_dim, feat_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(feat_dim, feat_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(feat_dim, eig_dim, rngs=rngs, use_bias=True),
        ])
        
        self.duals = nnx.Param(jnp.tril(duals_initial_val * jnp.ones((eig_dim, eig_dim))))

    def eig_conv(self, obs):
        obs = nnx.relu(self.conv1(obs))
        obs = nnx.max_pool(obs, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        obs = nnx.relu(self.conv2(obs))
        obs = nnx.max_pool(obs, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        obs = nnx.relu(self.conv3(obs))

        return obs

    def __call__(self, obs):
        if self.obs_type == 'image':

            obs = self.eig_conv(obs)
            obs = jnp.reshape(obs, (obs.shape[0], -1)) # flatten it
            eig_vec = self.image_linear(obs)
            return eig_vec
            
                            
        eig_vec = self.eig_linear(obs)
        return eig_vec