import numpy as np
import os
import gym
import matplotlib.pyplot as plt
from collections import deque
from flint import arb_mat, ctx

# testing imports
import gin
from drogo.reward_envs import maxent_mon_minigrid
from drogo.custom_wrappers import maxent_mdp_wrapper
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

ctx.dps = 100   # important
# print(ctx)

def symmetrize(M):
    return (M + M.T) / 2


directions =np.array([
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1]
])

class RewardShaper:
    """
    for reward shaping
    """


    def __init__(self, env):
        """
        env: environment
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
    

    
    def SR_top_eigenvector(self, pi=None, gamma=0.99):
        # make symmetric to ensure real eigenvectors and eigenvalues
        SR = symmetrize(self.compute_SR(pi=pi, gamma=gamma))

        lamb, e = np.linalg.eig(SR)
        idx = lamb.argsort()
        e = e.T[idx[::-1]]
        e0 = np.real(e[0]) 

        # normalize
        e0 /= np.sqrt(e0 @ e0)
        assert np.isclose(e0 @ e0, 1.0)

        if e0.sum() > 0:
            e0 *= -1

        return e0


    def compute_LR(self, pi=None, gamma=0.99, lambd=1.0):
        """
        Closed-form lambda-representation (Moskovitz et al., NeurIPS 2023) w.r.t. pi.
        If pi is None, use uniform random policy.
        lambd=1 recovers the SR; lambd=0 recovers the first-occupancy representation (FR).

        From the lambda-R Bellman fixed point, for each target state j:
            Phi[j, j] = 1 + gamma * lambd * (P_pi Phi)[j, j]
            Phi[s, j] = gamma * (P_pi Phi)[s, j]   for s != j
        i.e.  Phi[:, j] = (I - gamma * D_j P_pi)^{-1} e_j,
        with D_j = diag(1, ..., lambd at j, ..., 1).
        """
        if pi is None:
            pi = np.ones((self.env.num_states, self.env.num_actions)) / self.env.num_actions

        P = self.env.transition_probs
        P_pi = (P * pi[..., None]).sum(1)

        n = self.env.num_states
        I = np.eye(n)
        LR = np.zeros((n, n))

        for j in range(n):
            d = np.ones(n)
            d[j] = lambd
            # D_j @ P_pi scales the rows of P_pi by d
            A = I - gamma * (d[:, None] * P_pi)
            LR[:, j] = np.linalg.solve(A, I[:, j])

        return LR


    def LR_top_eigenvector(self, pi=None, gamma=0.99, lambd=1.0):
        """
        Top eigenvector of the (symmetrized) lambda-representation, extracted
        identically to SR_top_eigenvector. lambd=0 gives the FR baseline,
        lambd in (0, 1) the lambda-representation baseline.
        """
        LR = symmetrize(self.compute_LR(pi=pi, gamma=gamma, lambd=lambd))

        lamb, e = np.linalg.eig(LR)
        idx = lamb.argsort()
        e = e.T[idx[::-1]]
        e0 = np.real(e[0])

        # normalize
        e0 /= np.sqrt(e0 @ e0)
        assert np.isclose(e0 @ e0, 1.0)

        if e0.sum() > 0:
            e0 *= -1

        return e0


    def _grid_neighbors(self, s):
        """
        States adjacent to s on the 4-connected grid graph (walls excluded).
        Mirrors the neighbor enumeration in compute_neighboring_diff.
        """
        pos = self.env.state_to_pos[s]
        neighbors = []
        for dir in directions:
            neighbor_pos = pos + dir
            try:
                j = self.env.pos_to_state[neighbor_pos[0] + neighbor_pos[1] * self.env.width]
                if j >= 0:
                    neighbors.append(int(j))
            except Exception:
                pass
        return neighbors


    def distance_potential(self):
        """
        Distance-to-goal potential: phi(s) = -(shortest number of steps from s
        to the nearest goal) on the 4-connected grid graph, computed with BFS.
        Ignores rewards.
        """
        dist = np.full(self.env.num_states, np.inf)
        q = deque()
        for g in self.env.terminal_idx:
            dist[g] = 0
            q.append(int(g))

        while q:
            s = q.popleft()
            for j in self._grid_neighbors(s):
                if dist[j] == np.inf:
                    dist[j] = dist[s] + 1
                    q.append(j)

        return -dist


    def normalize(self, e):
        """
        Normalize 
        """
        assert len(e) == self.env.num_states

        # normalize so that mean of reward diff of neighboring states is 1.
        e /= np.mean(self.compute_neighboring_diff(e))
        assert np.isclose(np.mean(self.compute_neighboring_diff(e)), 1.0)

        return e
    
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
    

        
