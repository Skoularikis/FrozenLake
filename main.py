from frozen_lake import FrozenLake
from big_frozen_lake import big_frozen_lake
import numpy as np
import random


def play(env):
    # up, left, bottom, right
    actions = ['w', 'a', 's', 'd']

    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')

        state, r, done = env.step(actions.index(c))

        env.render()
        print('Reward: {0}'.format(r))


################ Model-based algorithms ################

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)

    for _ in range(max_iterations):
        delta = 0
        for s in range(env.n_states):
            v = value[s]
            value[s] = sum(
                [env.p(next_s, s, policy[s]) * (env.r(next_s, s, policy[s]) + gamma * value[next_s]) for next_s in
                 range(env.n_states)])

            delta = max(delta, abs(v - value[s]))
        if delta < theta:
            break
    return value


def policy_improvement(env, value, gamma, policy):
    policy_stable = True
    for s in range(env.n_states):
        pol = policy[s].copy()
        v = [
            sum([env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]) for next_s in range(env.n_states)])
            for a in range(env.n_actions)]
        policy[s] = np.argmax(v)
        if pol != policy[s]:
            policy_stable = False
    return policy_stable


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    value = np.zeros(env.n_states, dtype=int)
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    policy_stable = False
    index = 0
    while not policy_stable:
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy_stable = policy_improvement(env, value, gamma, policy)
        index += 1
    print(index)
    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    index = 0
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
    for _ in range(max_iterations):
        delta = 0.
        for s in range(env.n_states):
            v = value[s]
            value[s] = max([sum(
                [env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]) for next_s in range(env.n_states)])
                for a in range(env.n_actions)])

            delta = max(delta, np.abs(v - value[s]))

        if delta < theta:
            break

        index = index + 1

    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        policy[s] = np.argmax([sum(
            [env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]) for next_s in range(env.n_states)]) for
            a in range(env.n_actions)])

    print(index)
    return policy, value


################ Tabular model-free algorithms ################

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None, optimal_value=None, find_episodes=False):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        terminal = False
        a = e_greedy(q[s], epsilon[i], env.n_actions, random_state)

        # Select action a for state s according to an e-greedy policy based on Q. by random
        while not terminal:
            next_s, r, terminal = env.step(a)
            next_a = e_greedy(q[next_s], epsilon[i], env.n_actions, random_state)
            q[s][a] = q[s][a] + eta[i] * (r + (gamma * q[next_s][next_a]) - q[s][a])
            s = next_s
            a = next_a

        if find_episodes:
            value_new = policy_evaluation(env, q.argmax(axis=1), gamma, theta=0.001, max_iterations=200)
            if all(abs(optimal_value[i] - value_new[i]) < 0.001 for i in range(len(value_new))):
                print('Episodes to find the optimal policy: ' + str(i))
                return q.argmax(axis=1), value_new

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None, optimal_value=None, find_episodes=False):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    q = np.zeros((env.n_states, env.n_actions))
    for i in range(max_episodes):
        s = env.reset()
        terminal = False
        while not terminal:
            a = e_greedy(q[s], epsilon[i], env.n_actions, random_state)
            next_s, r, terminal = env.step(a)
            next_a = np.argmax(q[next_s])
            q[s][a] = q[s][a] + eta[i] * (r + (gamma * q[next_s][next_a]) - q[s][a])
            s = next_s

        if find_episodes:
            value_new = policy_evaluation(env, q.argmax(axis=1), gamma, theta=0.001, max_iterations=200)
            if all(abs(optimal_value[i]-value_new[i]) < 0.001 for i in range(len(value_new))):
                print('Episodes to find the optimal policy: ' + str(i))
                return q.argmax(axis=1), value_new

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


################ Non-tabular model-free algorithms ################

class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)
        a = e_greedy(q, epsilon[i], env.n_actions, random_state)
        terminal = False
        while not terminal:
            next_s, r, terminal = env.step(a)
            delta = r - q[a]
            q = next_s.dot(theta)
            a_new = e_greedy(q, epsilon[i], env.n_actions, random_state)

            delta = delta + (gamma * max(q))
            theta = theta + eta[i] * delta * features[a]
            features = next_s
            a = a_new

    return theta


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)
        terminal = False

        while not terminal:
            a = e_greedy(q, epsilon[i], env.n_actions, random_state)
            next_s, r, terminal = env.step(a)
            delta = r - q[a]
            q = next_s.dot(theta)
            delta = delta + (gamma * max(q))
            theta = theta + (eta[i] * delta * features[a])
            features = next_s
    return theta


def e_greedy(q, epsilon, n_actions, random_state):
    if random.uniform(0, 1) < epsilon:
        a = random_state.choice(np.flatnonzero(q == q.max()))
    else:
        a = random_state.randint(n_actions)
    return a


def optimal_policy_episodes(env, q, optimal_value, episode, gamma, theta=0.001, max_iterations=200):
    value_new = policy_evaluation(env, q.argmax(axis=1), gamma, theta, max_iterations)
    if all(abs(optimal_value[i] - value_new[i]) < 0.001 for i in range(len(value_new))):
        print('Episodes to find the optimal policy: ' + str(episode))
        return q.argmax(axis=1), value_new


def main():
    seed = 0

    # Small lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    big_lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '#', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '#', '#', '.', '.', '.', '#', '.'],
                ['.', '#', '.', '.', '#', '.', '#', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    # env = big_frozen_lake(big_lake, slip=0.1, max_steps=64, seed=seed)
    # play(env)

    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 100

    print('')

    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    opt_value = value.copy()

    print('')

    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')

    print('# Model-free algorithms')
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5

    print('')

    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed, optimal_value=opt_value,
                          find_episodes=True)
    env.render(policy, value)

    print(value)
    print(opt_value)
    print('')

    print('## Q-learning')
    policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed, optimal_value=opt_value,
                               find_episodes=True)
    env.render(policy, value)

    print('')

    print(value)
    print(opt_value)

    #
    #
    # linear_env = LinearWrapper(env)
    #
    # print('## Linear Sarsa')
    #
    # parameters = linear_sarsa(linear_env, max_episodes, eta,
    #                           gamma, epsilon, seed=seed)
    # policy, value = linear_env.decode_policy(parameters)
    # linear_env.render(policy, value)
    #
    # print('')
    #
    # print('## Linear Q-learning')
    #
    # parameters = linear_q_learning(linear_env, max_episodes, eta,
    #                                 gamma, epsilon, seed=seed)
    # policy, value = linear_env.decode_policy(parameters)
    # linear_env.render(policy, value)


if __name__ == "__main__":
    main()
