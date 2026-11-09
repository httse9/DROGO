import numpy as np
import os
import gym
import matplotlib.pyplot as plt
from flint import arb_mat, ctx

# testing imports
from drogo.examples.rep_utils import *
import gin
from drogo.reward_envs import maxent_mon_minigrid
from drogo.custom_wrappers import maxent_mdp_wrapper
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

ctx.dps = 100   # important
# print(ctx)

def sym(M):
    return (M + M.T) / 2


directions =np.array([
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1]
])

class RewardShaper:
    """
    Takes environment, compute 
    1. successor representation
    2. default representation
    3. maximum entropy representation

    4. shaping reward for SR
    5. shaping reward for DR (diff variants)
    6. shaping reward for MER (diff variants)
    """


    def __init__(self, env):
        """
        env: environment
        gamma: discount factor for SR
        lamb: lambda for DR and MER
        """
        self.env = env


    def compute_SR(self, pi = None, gamma=0.99):
        """
        Compute SR with respect to pi.
        If pi is None, use uniform random policy.
        """
        
        if pi is None:
            pi = np.ones((self.env.num_states, self.env.num_actions)) / self.env.num_actions

        P = self.env.transition_probs

        P_pi = (P * pi[..., None]).sum(1)

        SR = np.linalg.inv(np.eye(self.env.num_states) - gamma * P_pi)

        return SR.copy()
    
    def compute_DR(self, pi = None, lambd=1.0):
        """
        Compute DR with respect to pi.
        If pi is None, use uniform random policy.
        """
        if pi is None:
            pi = np.ones((self.env.num_states, self.env.num_actions)) / self.env.num_actions

        R = self.env.rewards
        P = self.env.transition_probs

        P_pi = (P * pi[..., None]).sum(1)
        DR = np.linalg.inv(np.diag(np.exp(-R / lambd)) - P_pi)

        return DR.copy()

    def compute_DR_2(self, pi = None, lambd=1.0):
        """
        Compute DR with respect to pi.
        If pi is None, use uniform random policy.
        """
        if pi is None:
            pi = np.ones((self.env.num_states, self.env.num_actions)) / self.env.num_actions

        R = self.env.rewards
        P = self.env.transition_probs

        P_pi = (P * pi[..., None]).sum(1)
        P_pi = sym(P_pi)
        DR = np.linalg.inv(np.diag(np.exp(-R / lambd)) - P_pi)

        return DR.copy()
    
    def SR_top_eigenvector(self, pi=None, gamma=0.99):
        # make symmetric to ensure real eigenvectors and eigenvalues
        SR = sym(self.compute_SR(pi=pi, gamma=gamma))

        lamb, e = np.linalg.eig(SR)
        idx = lamb.argsort()
        e = e.T[idx[::-1]]
        e0 = np.real(e[0]) 

        # normalize
        e0 /= np.sqrt(e0 @ e0)
        assert np.isclose(e0 @ e0, 1.0)

        return e0

    
    # def log_DR_top_eigenvector(self, pi=None, lambd=1.0):
    #     """
    #     Compute the top eigenvector for the log of the DR.
    #     """
    #     # make symmetric to ensure real eigenvectors and eigenvalues
    #     DR = sym(self.compute_DR(pi=pi, lambd=lambd))

    #     # take log
    #     DR = np.log(DR)
    #     # DR = -1 / DR

    #     # eigendecomposition & get top eigenvector
    #     lamb, e = np.linalg.eig(DR)
    #     idx = lamb.argsort()
    #     e = e.T[idx[::-1]]
    #     e0 = np.real(e[0])

    #     if e0[0] < 0:
    #         e0 *= -1

    #     return e0
    
    def DR_top_log_eigenvector(self, pi=None, lambd=1.0, normalize=True, symmetrize=True):
        """
        Compute log of the top eigenvector of the DR.
        """
        DR = self.compute_DR(pi=pi, lambd=lambd)
        if symmetrize:
            DR = sym(DR)
        assert (DR >= 0).all()

        DR = arb_mat(DR.tolist())
        lamb, e = DR.eig(right=True, algorithm="approx", )
        lamb = np.array(lamb).astype(np.clongdouble).real.flatten()
        e = np.array(e.tolist()).astype(np.clongdouble).real.astype(np.float32)

        idx = np.flip(lamb.argsort())
        lamb = lamb[idx]
        e = e.T[idx]
        e0 = e[0]

        # assert all entries are positive before taking log
        if e0[0] < 0:
            e0 *= -1
        assert (e0 > 0).all()

        log_e0 = np.log(e0)

        if normalize:
            log_e0 /= np.sqrt(log_e0 @ log_e0)
            assert np.isclose(log_e0 @ log_e0, 1.0)

        return log_e0

    
    def DR_top_log_eigenvector_2(self, pi=None, lambd=1.0, normalize=True, symmetrize=True):
        """
        Compute log of the top eigenvector of the DR.
        """
        DR = self.compute_DR_2(pi=pi, lambd=lambd)
        assert (DR >= 0).all()

        DR = arb_mat(DR.tolist())
        lamb, e = DR.eig(right=True, algorithm="approx", )
        lamb = np.array(lamb).astype(np.clongdouble).real.flatten()
        e = np.array(e.tolist()).astype(np.clongdouble).real.astype(np.float32)

        idx = np.flip(lamb.argsort())
        lamb = lamb[idx]
        e = e.T[idx]
        e0 = e[0]

        # assert all entries are positive before taking log
        if e0[0] < 0:
            e0 *= -1
        assert (e0 > 0).all()

        log_e0 = np.log(e0)

        if normalize:
            log_e0 /= np.sqrt(log_e0 @ log_e0)
            assert np.isclose(log_e0 @ log_e0, 1.0)

        return log_e0
    
    def shaping_reward_transform_using_terminal_state(self, e):
        """
        Transform input vector e into shaping reward:
        - | e(terminal state) - e(state) |
        """
        assert len(e) == self.env.num_states
        # idx of terminal state
        terminal_idx = self.env.terminal_idx[0]

        # compute shaping reward
        shaping_reward = - np.abs(e - e[terminal_idx])

        # normalize so that mean of reward diff of neighboring states is 1.
        shaping_reward /= np.mean(self.compute_neighboring_diff(shaping_reward))
        assert np.isclose(np.mean(self.compute_neighboring_diff(shaping_reward)), 1.0)

        return shaping_reward
    
    def compute_neighboring_diff(self, v):
        """
        Compute the difference of neighboring states of the vector v.
        """
        diffs = []

        for i in range(self.env.num_states):    # enumerate all states
            pos = self.env.state_to_pos[i]

            for dir in directions:      # enumerate all neighboring states
                neighbor_pos = pos + dir

                try:
                    j = self.env.pos_to_state[neighbor_pos[0] + neighbor_pos[1] * self.env.width]
                    if j >= 0:      # compute difference if neighbor exists
                        diffs.append(np.abs(v[i] - v[j]))
                except:
                    pass

        return diffs

        