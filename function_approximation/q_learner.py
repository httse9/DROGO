import numpy as np
import gym
import matplotlib.pyplot as plt
import os

# testing imports
import gin
from drogo.reward_envs import maxent_mon_minigrid
from drogo.custom_wrappers import maxent_mdp_wrapper
from drogo.function_approximation.reward_shaper import RewardShaper
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from drogo.examples.visualizer import Visualizer


class AuxiliaryReward:
    """
    """

    def __init__(self, env, r_aux, mode, r_aux_weight, gamma=0.99):
        """
        env: environment
        gamma: discount factor
        r_aux: auxiliary reward
        mode: potential, or none
        r_aux_weight: weight put on auxiliary reward

        Try not to modify r_aux...
        """
        self.env = env
        self.gamma = gamma
        self.r_aux = r_aux
        self.mode = mode
        assert 0 <= r_aux_weight <= 1
        self.r_aux_weight = r_aux_weight


    def shaped_reward(self, r_orig, state, next_state):
        """
        Return shaped reward based on the mode
        """
        if self.mode == "none":
            return r_orig
        elif self.mode == "potential":
            return self.potential_shaped_reward(r_orig, state, next_state)


    def potential_shaped_reward(self, r_orig, state, next_state):
        """
        Potential-based reward shaping

        Params:
        r_orig: reward of the original MDP
        state: scalar, the current state
        next state: scalar, the next state
        """
        # average of r_orig and potential reward of r_aux
        return (1 - self.r_aux_weight) * r_orig + self.r_aux_weight * (self.gamma * self.r_aux[next_state] - self.r_aux[state])


class QLearner:
    """
    Run Q-Learning
    """
    def __init__(self, env, env_eval, aux_reward: AuxiliaryReward, step_size, gamma=0.99):
        """
        env: environment
        aux_reward: AuxiliaryReward object to provide auxiliary rewards
        """
        self.env = env
        self.env_eval = env_eval
        self.aux_reward = aux_reward
        self.step_size = step_size
        self.gamma = gamma

        # init Q values
        self.Q = np.zeros((env.num_states, env.num_actions))

    def epsilon_greedy_action_selection(self, state):
        """
        state: scalar
        epsilon: 0.05
        """
        if np.random.rand() < 0.05: # epsilon greedy
            a = np.random.choice(self.env.num_actions)
        else:
            a = np.argmax(self.Q[state])
        return a
    
    def update_Q(self, s, a, r, ns, terminated):
        """
        Update Q value given transition
        """
        if terminated:
            self.Q[s, a] += self.step_size * (r - self.Q[s, a])
        else:
            self.Q[s, a] += self.step_size * (r + self.gamma * self.Q[ns].max() - self.Q[s, a])

    def get_current_policy(self):
        """
        Get policy from current Q values
        """
        pi = (self.Q == self.Q.max(1, keepdims=True)).astype(float)
        pi /= pi.sum(1, keepdims=True)
        return pi

    def eval_policy(self):
        """
        Evaluate current policy and return undiscounted return.
        Env is deterministic, policy also (not exactly but shouldn't matter much). One episode suffices.
        """
        pi = self.get_current_policy()

        env = self.env_eval
        s = env.reset()
        ret = 0     # return

        done = False
        while not done:
            a = np.random.choice(env.num_actions, p=pi[s['state']])
            s, r, done, _ = env.step(a)
            ret += r

        return ret

    def learn(self, max_iter, log_interval):
        """
        Learn.
        """

        env = self.env
        s = env.reset()

        timesteps = [0]
        returns = [self.eval_policy()]
        Qs = [self.Q.copy()]

        visit_low_reward = 0

        for n in range(max_iter):
            # choose action
            a = self.epsilon_greedy_action_selection(s['state'])

            # env step
            ns, r, done, d = env.step(a)
            terminated = d['terminated']

            visit_low_reward += (r == -20)

            # compute shaped reward
            shaped_reward = self.aux_reward.shaped_reward(r, s['state'], ns['state'])

            # update Q
            self.update_Q(s['state'], a, shaped_reward, ns['state'], terminated)

            # reset environment if needed
            if done:
                s = env.reset()
            else:
                s = ns

            # eval policy regularly
            if (n + 1) % log_interval == 0:
                timesteps.append(n + 1)
                returns.append(self.eval_policy())
                # print(returns[-1])

                if (n + 1) % 1000 == 0:
                    Qs.append(self.Q.copy())

        return np.array(timesteps), np.array(returns), np.array(Qs), visit_low_reward


def eigvec_myopic_policy(env, eigvec):
    """
    Get the myopic (hill-climbing policy) for current eigenvector
    """
    termination = np.zeros((env.num_states))
    policy = np.zeros((env.num_states))

    for s in range(env.num_states):

        # handle unvisited state / terminal state
        if s in env.terminal_idx:
            termination[s] = 1
            continue

        # for visited states:
        pos = env.state_to_pos[s]  # (x, y): x-th col, y-th row
        value = eigvec[s]  # init value
        myopic_a = -1

        for a, dir_vec in enumerate(np.array([
            [1, 0], # right
            [0, 1], # down
            [-1, 0],    # left
            [0, -1],    # up
        ])):
            neighbor_pos = pos + dir_vec
            neighbor_state = env.pos_to_state[neighbor_pos[0] + neighbor_pos[1] * env.width]
            
            # if neighbor state exists (not wall) 
            # and neighor state has been visited
            # and has higher eigenvector value
            # go to that neighbor state
            if neighbor_state >= 0 and eigvec[neighbor_state] > value:
                value = eigvec[neighbor_state]
                myopic_a = a

        if myopic_a == -1:
            # no better neighbor, terminate
            termination[s] = 1
        else:
            policy[s] = myopic_a

    myopic_policy = dict(termination=termination, policy=policy)
    return myopic_policy
            
