"""
Generate dataset by 
1) sampling a random state uniformly
2) sampling a random action uniformly
and 
3) record different types of observation.
    - one-hot
    - coordinates
    - image
For each transition, record
- s, a, r, s', r', terminated (whether s' is terminal state)


When not using anchoring, meaning we use quadratic norm penalty,
we sample the terminal state as well.
"""

import gym
import gin
from drogo.reward_envs import maxent_mon_minigrid
from drogo.custom_wrappers import maxent_mdp_wrapper
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import numpy as np
import pickle
from os.path import join
from copy import deepcopy

def generate_dataset(env, N, terminal_s):

    dataset = {
        "onehot": [],
        "coordinates": [],
        "image": []
    }

    for _ in range(N):

        s = env.reset()     # samples a random start state
        one_hot, coordinates, image = get_observations(env, s)

        a = np.random.choice(env.num_actions)
        ns, r, done, d = env.step(a)
        nr = env.reward()
        terminated = float(ns['state'] in env.terminal_idx)

        if s['state'] == terminal_s:
            ns = deepcopy(s)
            r = -0.001
            nr = r
            terminated = 1.

        one_hot_next, coordinates_next, image_next = get_observations(env, ns)

        dataset['onehot'].append([one_hot, a, r, one_hot_next, nr, terminated])
        dataset['coordinates'].append([coordinates, a, r, coordinates_next, nr, terminated])
        dataset['image'].append([image, a, r, image_next, nr, terminated])

    return dataset

def create_test_set(env):
    """
    Create a dataset consisting of all states/observations in an environment
    For testing the learned eigenvector
    """

    dataset = {
        "onehot": [],
        "coordinates": [],
        "image": []
    }
    
    # enumerate over all states
    for s in range(env.num_states):

        env.reset()
        x, y = env.state_to_pos[s]
        # set start position to s
        env.unwrapped.agent_pos = [x, y]

        dataset['onehot'].append(onehot(env.num_states, s))
        dataset['coordinates'].append(np.array([x / (env.width - 1), y / (env.height - 1)]) - 0.5)
        dataset['image'].append(env.custom_rgb() - 0.5)

    return dataset



def get_observations(env, s):
    """
    s: the state returned by environment, not a singe scalar
    """
    one_hot = onehot(env.num_states, s['state'])

    x, y = env.agent_pos
    coordinates = np.array([x / (env.width - 1), y / (env.height - 1)]) - 0.5

    image = env.custom_rgb() - 0.5

    return one_hot, coordinates, image



def onehot(num_states, idx):
    v = np.zeros((num_states)) * 1.
    if idx is None:
        return v
    v[idx] = 1
    return v

        


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="fourrooms_2", type=str, help="Specify environment.")
    parser.add_argument("--num_transitions", default=200_000, type=int, help="Number of transitions to geenrate")
    args = parser.parse_args()


    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{args.env}.gin"))
    env_id = maxent_mon_minigrid.register_environment()


    env_dummy = gym.make(env_id, no_goal=False, no_start=True)    # random initial state, including terminal
    env_dummy = maxent_mdp_wrapper.MDPWrapper(env_dummy, goal_absorbing=True)
    terminal_s = env_dummy.terminal_idx[0]
    del env_dummy

    np.random.seed(1)
    env = gym.make(env_id, no_goal=True, no_start=True)    # random initial state, including terminal
    env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True)

    dataset_path = join("drogo", "function_approximation", "dataset")
    os.makedirs(dataset_path, exist_ok=True)

    dataset = generate_dataset(env, args.num_transitions, terminal_s)
    with open(join(dataset_path, f"{args.env}_dataset_qp.pkl"), "wb") as f:
        pickle.dump(dataset, f)

    test_set = create_test_set(env)
    with open(join(dataset_path, f"{args.env}_testset.pkl"), "wb") as f:
        pickle.dump(test_set, f)