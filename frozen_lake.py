import contextlib
from itertools import product

import numpy as np

from environment import Environment


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
         lake =  [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        # self.lake = np.zeros(np.array(lake).shape)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        self.slip = slip
        n_states = self.lake.size + 1
        n_actions = 4
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        self.absorbing_state = n_states - 1

        # TODO:
        Environment.__init__(self, n_states, 4, max_steps, pi, seed)

        # Up, left, down, right.
        self.actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        self.itos = list(product(range(self.lake.shape[0]), range(self.lake.shape[1])))
        self.stoi = {s: i for (i, s) in enumerate(self.itos)}

        self._p = np.zeros((n_states, n_states, 4))

        for state_index, state in enumerate(self.itos):
            for action_index, action in enumerate(self.actions):
                next_state = (state[0] + action[0], state[1] + action[1])
                if (state_index == 5 or state_index == 7 or state_index == 12 or state_index == 11 or state_index == 15):
                    self._p[state_index, state_index, action_index] = 1.0
                else:
                    next_state_index = self.stoi.get(next_state, state_index)
                    self._p[next_state_index, state_index, action_index] = 1 - self.slip
                    for act in self.actions:
                        next_state_action = (state[0] + act[0], state[1] + act[1])
                        next_state_index = self.stoi.get(next_state_action, state_index)
                        self._p[next_state_index, state_index, action_index] += self.slip/4

    def step(self, action):
        state, reward, done = Environment.step(self, action)
        done = (state == self.absorbing_state) or done
        return state, reward, done

    def p(self, next_state, state, action):
        # TODO:
        return self._p[next_state, state, action]

    def r(self, next_state, state, action):
        # TODO:
        if state == 15:
            return 1
        else:
            return 0
        # return self.lake[self.itos[state]]

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)
            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']

            print('Lake:')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            # print('Value:')
            # with self._printoptions(precision=3, suppress=True):
            #     print(value[:-1].reshape(self.lake.shape))

    # Configures numpy print options
    @contextlib.contextmanager
    def _printoptions(*args, **kwargs):
        original = np.get_printoptions()
        np.set_printoptions(*args, **kwargs)
        try:
            yield
        finally:
            np.set_printoptions(**original)