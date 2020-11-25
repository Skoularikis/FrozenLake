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

        # self.lake[1,1]=1
        # self.lake[3,0]=1
        # self.lake[1,3]=1
        # self.lake[2,3]=1
        # self.goal = self.lake[3,3] = 1

        # TODO:
        Environment.__init__(self, n_states, 4, max_steps, pi, seed)
        # # up, left, bottom, right
        # actions = ['w', 'a', 's', 'd']
        # Up, left, down, right.
        self.actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        self.itos = list(product(range(self.lake.shape[0]), range(self.lake.shape[1])))
        self.stoi = {s: i for (i, s) in enumerate(self.itos)}

        self._p = np.zeros((n_states, n_states, 4))

        for state_index, state in enumerate(self.itos):
            for action_index, action in enumerate(self.actions):
                next_state = (state[0] + action[0], state[1] + action[1])

                # If next_state is not valid, default to current state index
                next_state_index = self.stoi.get(next_state, state_index)
                # self._p[next_state_index, state_index, action_index] = 1.0
                if next_state_index == state_index:
                    self._p[next_state_index, state_index, action_index] = 1
                else:
                    self._p[next_state_index, state_index, action_index] = 0.9

            a = np.array(self._p[:,state_index])
            unique, counts = np.unique(self._p[:,state_index], return_counts=True)
            my_dict = dict(zip(unique, counts))
            for idx, ai in enumerate(a):
                if 0.9 in ai:
                    for qd in range(0, 4):
                        if 0.9 in a[:, qd] and a[:, qd][idx] == 0.0:
                            if (0.9 in my_dict):
                                a[:, qd][idx] = 0.1 / (my_dict[0.9] - 1)
            self._p[:,state_index] = a
        print(self._p[:,5])

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

            print('Value:')
            with self._printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))

    # Configures numpy print options
    @contextlib.contextmanager
    def _printoptions(*args, **kwargs):
        original = np.get_printoptions()
        np.set_printoptions(*args, **kwargs)
        try:
            yield
        finally:
            np.set_printoptions(**original)