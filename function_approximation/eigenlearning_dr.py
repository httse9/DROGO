import numpy as np
import jax.numpy as jnp
import random
import pickle
import gym
import gin
from drogo.reward_envs import maxent_mon_minigrid
from drogo.custom_wrappers import maxent_mdp_wrapper
from drogo.examples.visualizer import Visualizer
import matplotlib.pyplot as plt
import os
from os.path import join
from drogo.function_approximation.eigenlearner_FA import *
import wandb

## prevent numpy.linalg.inv from hanging........... ( -_-'') ?
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

learners = {
    "drogo": DROGO_Learner,
    "natgrad": NATGRAD_Learner,
    "gdo": GDO_Learner,
    "log": LOG_Learner
}

def eigenlearning(args, env):

    ### load dataset
    if args.dataset == "static":
        dataset, test_set = load_static_dataset(args)
    elif args.dataset == "real":
        dataset, test_set = load_dataset(args)

    ### for visualizing eigvec
    visualizer = Visualizer(env)
    cmap = "rainbow"  

    ### init learner
    learner = learners[args.dr_mode](env, dataset, test_set, args)
    learner.init_learn()

    ### learn
    learner.learn()

    ### plotting
    eigvec = learner.eigvec()

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    visualizer.visualize_shaping_reward_2d(learner.true_eigvec, ax=None, normalize=True, vmin=0, vmax=1, cmap=cmap)
    plt.subplot(1, 2, 2)
    visualizer.visualize_shaping_reward_2d(eigvec, ax=None, normalize=True, vmin=0, vmax=1, cmap=cmap)
    plt.tight_layout()
    plt.savefig(join(plot_path, f"{run_name}_eigvec.png"))
    wandb.log({"eigvec_vis": wandb.Image(plt)})
    plt.clf()

    ### save raw data
    data = dict(
        cos_sims=learner.cos_sims,
        norms=learner.norms,
        true_eigvec=learner.true_eigvec,
        eigvec=eigvec
    )
    with open(join(data_path, f"{run_name}.pkl"), "wb") as f:
        pickle.dump(data, f)

### Ground-truth small dataset (for testing)
def load_static_dataset(args):
    if args.dr_mode == "drogo":
        with open(f"drogo/function_approximation/static_dataset/{args.env}_{args.obs_type}_2.pkl", "rb") as f:
            dataset = pickle.load(f)
    else:
        with open(f"drogo/function_approximation/static_dataset/{args.env}_{args.obs_type}.pkl", "rb") as f:
            dataset = pickle.load(f)

    with open(f"drogo/function_approximation/static_dataset/{args.env}_{args.obs_type}_test.pkl", "rb") as f:
        test_set = jnp.array(pickle.load(f))

    return dataset, test_set

### Actual dataset used in paper
def load_dataset(args):

    if args.dr_mode == "drogo":
        with open(f"drogo/function_approximation/dataset/{args.env}_dataset.pkl", "rb") as f:
            dataset = pickle.load(f)[args.obs_type]
    else:
        with open(f"drogo/function_approximation/dataset/{args.env}_dataset_qp.pkl", "rb") as f:
            dataset = pickle.load(f)[args.obs_type]

    with open(f"drogo/function_approximation/dataset/{args.env}_testset.pkl", "rb") as f:
        test_set = jnp.array(pickle.load(f)[args.obs_type])

    return dataset, test_set

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="fourrooms_2", type=str, help="Specify environment.")
    parser.add_argument("--obs_type", default="onehot", type=str)
    parser.add_argument("--dataset", default="real", type=str)

    parser.add_argument("--dr_mode", type=str, default="drogo")
    parser.add_argument("--optimizer", default="rmsprop", type=str)
    
    parser.add_argument("--lambd", help="lambda for DR", default=20.0, type=float)
    parser.add_argument("--step_size_start", default=1e-5, type=float, help="Starting step size")
    parser.add_argument("--step_size_end", default=None, type=float, help="Ending step size. Default: None, meaning equal to start")
    parser.add_argument("--grad_norm_clip", default=0.5, type=float, help="Ending step size")

    parser.add_argument("--n_epochs", type=int, default=10000, help="Number of passes thru dataset")
    parser.add_argument("--batch_size", default=2000, help="Batch size", type=int)

    parser.add_argument("--eig_dim", type=int, default=1, help="How many dimension of laplacian representation to learn")
    parser.add_argument("--barrier", type=float, default=0.5, help="Barrier coefficient b. Not used for anchor")
    
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.step_size_end is None:
        args.step_size_end = args.step_size_start

    if args.dr_mode == "drogo":
        args.barrier = 0

    # create env
    set_random_seed(args.seed)
    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{args.env}.gin"))
    env_id = maxent_mon_minigrid.register_environment()
    env = gym.make(env_id, disable_env_checker=True)
    env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True, goal_absorbing_reward=-0.001)


    # create dir for saving results
    path = join("drogo", "function_approximation", f"experiments_{args.dr_mode}_{args.dataset}", args.env, args.obs_type)
    plot_path = join(path, "plots")
    data_path = join(path, "data")
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    group_name = f"{args.lambd}-{args.step_size_start}-{args.step_size_end}-{args.batch_size}-{args.barrier}-{args.optimizer}-{args.dr_mode}-{args.grad_norm_clip}"
    run_name = group_name + f"-{args.seed}"

    run = wandb.init(
        project=f"testing",
        config=vars(args),
        group=group_name,  
        job_type="train",
        name=run_name, 
    )
    run.define_metric("train/*", step_metric="train/epoch")

    # learn
    eigenlearning(args, env)

    run.finish()



    