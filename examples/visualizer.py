import numpy as np
import os
import gym
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# testing imports
import gin
from drogo.reward_envs import maxent_mon_minigrid
from drogo.custom_wrappers import maxent_mdp_wrapper
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

action_markers = {
    0: '>',
    1: 'v',
    2: '<',
    3: '^'
}


class Visualizer:
    """
    Visualize environment, reward, and etc.
    """

    def __init__(self, env):
        """
        env: environment
        """
        self.env = env

    def visualize_env(self,):
        """
        Visualize environment
        """
        grid = self.env.raw_grid.T
        h, w = grid.shape
        image = np.ones((h, w, 3))

        for i in range(h):
            for j in range(w):
                if grid[i, j] == '*':
                    # wall
                    image[i, j] = np.array((44, 62, 80)) / 255.  # gray

                elif grid[i, j] == 'l':
                    # lava
                    image[i, j] = np.array((231, 76, 60)) / 255.    # orange

                elif grid[i, j] == 's':
                    # agent
                    image[i, j] = np.array((41, 128, 185)) / 255.   # blue

                elif grid[i, j] == 'g':
                    # goal
                    image[i, j] = np.array((46, 204, 113)) / 255. # green

        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout()


    def visualize_shaping_reward_2d(self, reward, ax=None, normalize=True, vmin=0, vmax=1, cmap="Reds"):

        if ax is None:
            ax = plt.gca()

        grid = self.env.raw_grid.T
        h, w = grid.shape
        image = np.ones((h, w))

        # normalize reward
        if normalize:
            reward_normalized = reward - reward.min()
            if not (reward_normalized == 0).all():
                reward_normalized /= reward_normalized.max()
            assert (reward_normalized >= 0).all() and (reward_normalized <= 1).all()
        else:
            reward_normalized = reward

        # construct the map with reward
        # let walls have value 0 for now
        state_num = 0
        for i in range(h):
            for j in range(w):
                if grid[i, j] == '*':
                    # wall
                    image[i, j] = 0.
                    continue

                # set reward
                image[i, j] = reward_normalized[state_num]
                state_num += 1

        # use plt cmap to get colored image
        # cmap = plt.get_cmap('rainbow')
        cmap = plt.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
        image = scalar_map.to_rgba(image)

        # draw walls again
        for i in range(h):
            for j in range(w):
                if grid[i, j] == '*':
                    image[i, j, :3] = np.array((44, 62, 80)) / 255.
        

        ax.imshow(image)
        ax.axis('off')
        plt.tight_layout()
        return image

    def visualize_vec(self, vec, ax=None, center=0):
        """
        Visualize a vector while centering at center
        The center value will have neural color,
        while larger values have red, smaller values have blue
        """

        if ax is None:
            ax = plt.gca()

        grid = self.env.raw_grid.T
        h, w = grid.shape
        image = np.ones((h, w))

        # construct the map with reward
        # let walls have value 0 for now
        state_num = 0
        for i in range(h):
            for j in range(w):
                if grid[i, j] == '*':
                    # wall
                    image[i, j] = 0.
                    continue

                # set reward
                image[i, j] = vec[state_num]
                state_num += 1

        # use plt cmap to get colored image
        c1 = (0.03137254901960784, 0.18823529411764706, 0.4196078431372549, 1.0) #"#3498db"
        c2 = (0.403921568627451, 0.0, 0.05098039215686274, 1.0) #"#e74c3c"
        cmap = LinearSegmentedColormap.from_list("flat_coolwarm", [c1, "white", c2])    # custom flat color map
        norm = mpl.colors.CenteredNorm(vcenter=center)
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
        image = scalar_map.to_rgba(image)

        # draw walls again
        for i in range(h):
            for j in range(w):
                if grid[i, j] == '*':
                    image[i, j, :3] = np.array((44, 62, 80)) / 255.
        

        ax.imshow(image)
        ax.axis('off')
        plt.tight_layout()

    def visualize_option(self, option, ax=None):

        marker_size = 9.5

        if ax is None:
            ax = plt.gca()

        grid = self.env.raw_grid.T
        h, w = grid.shape
        image = np.ones((h, w, 3))

        # draw walls
        for i in range(h):
            for j in range(w):
                if grid[i, j] == '*':
                    # wall
                    image[i, j, :3] = np.array((44, 62, 80)) / 255.


        ax.imshow(image)

        for s, a, in enumerate(option['policy']):

            x, y = self.env.state_to_pos[s]

            # if termination, plot terminate sign
            if option['termination'][s]:
                ax.plot([x], [y], marker='o', markersize=marker_size, color='#c0392b')
                continue

            # if not termination set, plot action
            ax.plot([x ], [y ], marker=action_markers[a], markersize=marker_size, color="#1abc9c")
            
        ax.axis('off')
        plt.tight_layout()


    def visualize_option_with_env_reward(self, option, ax=None):
        marker_size = 9.5

        if ax is None:
            ax = plt.gca()

        grid = self.env.raw_grid.T
        h, w = grid.shape
        image = np.ones((h, w, 3))

        # draw walls
        for i in range(h):
            for j in range(w):
                if grid[i, j] == '*':
                    # wall
                    image[i, j, :3] = np.array((44, 62, 80)) / 255.
                elif grid[i, j] == 'l':
                    # lava
                    image[i, j] = np.array((231, 76, 60)) / 255.    # orange


        ax.imshow(image)

        for s, a, in enumerate(option['policy']):

            x, y = self.env.state_to_pos[s]

            # if termination, plot terminate sign
            if option['termination'][s]:
                ax.plot([x], [y], marker='o', markersize=marker_size, color='#c0392b')
                continue

            # if not termination set, plot action
            ax.plot([x ], [y ], marker=action_markers[a], markersize=marker_size, color="#1abc9c")
            
        ax.axis('off')
        plt.tight_layout()
        




