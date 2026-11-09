"""
Create a minimal fixed dataset with a uniform initial state distibution,
and a uniform policy.

The dataset size is |S| * |A|
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
from copy import deepcopy

def create_dataset(env, obs_type):
    """
    Create a dataset consisting of all transitions in an environment

    Each entry is (s, a, r, s', r', is_goal)
    s: state
    a: action
    r: reward
    s': next state
    r': next reward
    is_goal: whether current state, s, is terminal. Not s'!!
    """

    dataset = []    # (s, a, r, s', r', is_goal?)
    
    # enumerate over all states
    for s in range(env.num_states):

        env.reset()
        x, y = env.state_to_pos[s]
        # set start position to s
        env.unwrapped.agent_pos = [x, y]

        if obs_type == "onehot":
            curr_s = onehot(env.num_states, s)
        elif obs_type == "coordinates":
            curr_s = np.array([x / env.width, y / env.height]) - 0.5
            assert (curr_s <= 1).all()
        elif obs_type == "image":
            curr_s = env.custom_rgb() - 0.5
        elif obs_type == "state_num":
            curr_s = s

        terminated = float(s in env.terminal_idx)

        for a in range(env.num_actions):
            # set start position to s
            env.unwrapped.agent_pos = [x, y]
            
            # take action a
            ns, r, done, d = env.step(a)

            if terminated:
                # if terminal state, transition to itself with prob 1
                next_s = deepcopy(curr_s)
                next_r = r

            else:
                if obs_type == "onehot":
                    next_s = onehot(env.num_states, ns['state'])
                elif obs_type == "coordinates":
                    next_x, next_y = env.agent_pos
                    next_s = np.array([next_x / env.width, next_y / env.height]) - 0.5
                    assert (next_s <= 1).all()
                elif obs_type == "image":
                    next_s = env.custom_rgb() - 0.5
                elif obs_type == "state_num":
                    next_s = ns['state']

                next_r = env.reward()

            # record transition
            dataset.append((curr_s, a, r, next_s, next_r, terminated))

    return dataset


def create_dataset_2(env, obs_type):
    """
    Create a dataset consisting of all transitions in an environment
    Excluding transitions starting from terminal state
    terminal bit refers to whether s' is terminal, not s
    """

    dataset = []    # (s, a, r, s', is_s'_goal?)
    
    # enumerate over all states
    for s in range(env.num_states):

        if s in env.terminal_idx:   # skip terminal state
            continue

        env.reset()
        x, y = env.state_to_pos[s]
        # set start position to s
        env.unwrapped.agent_pos = [x, y]

        if obs_type == "onehot":
            curr_s = onehot(env.num_states, s)
        elif obs_type == "coordinates":
            curr_s = np.array([x / env.width, y / env.height]) - 0.5
            assert (curr_s <= 1).all()
        elif obs_type == "image":
            curr_s = env.custom_rgb() - 0.5
        elif obs_type == "state_num":
            curr_s = s

        for a in range(env.num_actions):
            # set start position to s
            env.unwrapped.agent_pos = [x, y]
            
            # take action a
            ns, r, done, d = env.step(a)

            if obs_type == "onehot":
                next_s = onehot(env.num_states, ns['state'])
            elif obs_type == "coordinates":
                next_x, next_y = env.agent_pos
                next_s = np.array([next_x / env.width, next_y / env.height]) - 0.5
                assert (next_s <= 1).all()
            elif obs_type == "image":
                next_s = env.custom_rgb() - 0.5
            elif obs_type == "state_num":
                next_s = ns['state']

            next_r = env.reward()

            # record transition
            dataset.append((curr_s, a, r, next_s, next_r, float(ns['state'] in env.terminal_idx)))

    return dataset

def create_test_set(env, obs_type):
    """
    Create a dataset consisting of all states/observations in an environment
    For testing the learned eigenvector
    """

    dataset = []    # (s, a, r, s', is_goal?)
    
    # enumerate over all states
    for s in range(env.num_states):

        env.reset()
        x, y = env.state_to_pos[s]
        # set start position to s
        env.unwrapped.agent_pos = [x, y]

        if obs_type == "onehot":
            curr_s = onehot(env.num_states, s)
        elif obs_type == "coordinates":
            curr_s = np.array([x / env.width, y / env.height]) - 0.5
            assert (curr_s <= 1).all()
        elif obs_type == "image":
            curr_s = env.custom_rgb() - 0.5
        elif obs_type == "state_num":
            curr_s = s

        dataset.append(curr_s)

    return dataset


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
    parser.add_argument("--obs_type", default="onehot", type=str, help="Type of environment observation")
    args = parser.parse_args()

    if args.obs_type not in ['onehot', 'coordinates', 'image', "state_num"]:
        raise NotImplementedError()

    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{args.env}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    np.random.seed(1)
    env = gym.make(env_id, seed=0, no_goal=False)
    env = maxent_mdp_wrapper.MDPWrapper(env, goal_absorbing=True)

    dataset = create_dataset(env, args.obs_type)
    with open(f"drogo/function_approximation/static_dataset/{args.env}_{args.obs_type}.pkl", "wb") as f:
        pickle.dump(dataset, f)

    # print(dataset)

    dataset = create_dataset_2(env, args.obs_type)
    with open(f"drogo/function_approximation/static_dataset/{args.env}_{args.obs_type}_2.pkl", "wb") as f:
        pickle.dump(dataset, f)

    test_set = create_test_set(env, args.obs_type)
    with open(f"drogo/function_approximation/static_dataset/{args.env}_{args.obs_type}_test.pkl", "wb") as f:
        pickle.dump(test_set, f)