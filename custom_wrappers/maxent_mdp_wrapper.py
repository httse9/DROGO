# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper to provide access to the MDP objects (namely, `transition_probs` and `rewards`).

This extends the TabularWrapper, which converts a MiniGridEnv to a tabular MDP.

This iterates through the states (assuming the number of states is given by
width * height) and constructs the transition probabilities. It assumes, like
in the stochasticity setting of our local copy of MiniGridEnv, that the
stochasticity probability mass is distributed evenly across the three actions
different from the one chosen, and staying in-place.
"""

import numpy as np
from drogo.custom_wrappers import tabular_wrapper


def get_next_state(pos, env):
  """Return the next state.

  Args:
    pos: pair of ints, current agent position.
    env: MiniGrid environment.

  Returns:
    The next agent position.
  """
  # get next state
  next_pos = pos
  fwd_pos = env.front_pos
  cell = env.grid.get(*fwd_pos)
  if cell is None or cell.can_overlap():
    next_pos = fwd_pos

  return next_pos


class MDPWrapper(tabular_wrapper.TabularWrapper):
  """Wrapper to provide access to the MDP objects (namely, `transition_probs` and `rewards`).
  """

  def __init__(self, env, tile_size=8, get_rgb=False, goal_absorbing=False, goal_absorbing_reward=0):
    """
    goal_absorbing: whether the goal states are absorbing. If True, goal state
      transition to itself with prob 1. If False, goal state transition to 
      the absorbing state with prob 1, and the entries in the transition prob
      matrix corresponding to the goal states are all 0.
    """
    super().__init__(env, tile_size=tile_size, get_rgb=get_rgb)

    assert self.stochasticity == 0

    self.num_actions = len(env.actions)
    self.transition_probs = np.zeros((self.num_states, self.num_actions,
                                      self.num_states))
    self.rewards = np.zeros((self.num_states, ))    # reward only depends on state
    env = self.unwrapped

    self.terminal_idx = []

    # state numbers are ordered starting from 1st row (left to right), 
    # then 2nd row, then ...
    for y in range(self.height):
      for x in range(self.width):

        s1 = self.pos_to_state[x + y * env.width]

        if s1 < 0:  # Invalid position. (ignore walls)
          continue

        cell = self.grid.get(x, y)

        # take care of goal states
        if cell is not None and cell.type == 'goal':
          # ignore transition, cause transition prob from goal is all zeros
          # set reward of goal states

          self.terminal_idx.append(s1)
          self.rewards[s1] = self.reward_dict[env._raw_grid[x, y]] 

          if goal_absorbing:
            """
            Note that the changes here only affect the transition prob matrix(?) and reward vector
            that we construct for the sake of computing the DR, and does not affect the underlying MDP,
            which still treats the goal as not absorbing.

            We are changing how we formulate the DR, which is a part of the solution, and we are not
            altering the problem setting in which we are operating, which is why the terminal state
            is still non-absorbing, and has reward of 0.
            """
            # if treat goal as absorbing state
            for a in range(self.num_actions):
              self.transition_probs[s1, a, s1] = 1.

            # set reward to 0 or very tiny negative for the corrected DR
            self.rewards[s1] = goal_absorbing_reward 
            self.reward_grid[x, y] = goal_absorbing_reward

          continue

        # non-goal states
        env.agent_pos = np.array([x, y])
        r = env._reward() # pylint: disable=protected-access
        for a in range(self.num_actions):
          env.agent_dir = a
          next_pos = get_next_state([x, y], env)
          s2 = self.pos_to_state[next_pos[0] + next_pos[1] * env.width]
          self.transition_probs[s1, a, s2] = 1.
          self.rewards[s1] = r

    self.nonterminal_idx = (self.transition_probs.sum(-1).sum(-1) != 0)

  def set_start_state(self, start_state):
    x, y = self.state_to_pos[start_state]
    self.env.raw_grid[x, y] = 's'

  def custom_rgb(self):
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

            ### self.env.raw_grid stores the initial configuration of the environment
            ### We need to read the current position of the agent using self.env.agent_pos (see below after for loop)
            # elif grid[i, j] == 's':
            #     # agent
            #     image[i, j] = np.array((41, 128, 185)) / 255.   # blue

            elif grid[i, j] == 'g':
                # goal
                image[i, j] = np.array((46, 204, 113)) / 255. # green

    # read current agent position and draw agent
    y, x = self.unwrapped.agent_pos
    image[x, y] = np.array((41, 128, 185)) / 255.

    return image
