from frozen_lake import FrozenLake

BIG_LAKE_ROWS = 8
BIG_LAKE_COLS = 8

SMALL_LAKE_ROWS = 4
BIG_LAKE_ROWS = 4

BIG_LAKE_GOAL = (7,7)
SMALL_LAKE_GOAL = (3,3)

START = (0, 0)

SMALL_LAKE_ABS = [(1,1),(1,3),(2,3),(3,0)]
BIG_LAKE_ABS = [(2,3),(3,5),(4,3),(5,1),(5,2),(5,6),(6,1),(6,4),(6,6),(7,3)]





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

def main():
    seed = 0

    # Small lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)

    play(env)

    # print('# Model-based algorithms')
    # gamma = 0.9
    # theta = 0.001
    # max_iterations = 100
    #
    # print('')
    #
    # print('## Policy iteration')
    # policy, value = policy_iteration(env, gamma, theta, max_iterations)
    # env.render(policy, value)
    #
    # print('')
    #
    # print('## Value iteration')
    # policy, value = value_iteration(env, gamma, theta, max_iterations)
    # env.render(policy, value)
    #
    # print('')
    #
    # print('# Model-free algorithms')
    # max_episodes = 2000
    # eta = 0.5
    # epsilon = 0.5
    #
    # print('')
    #
    # print('## Sarsa')
    # policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    # env.render(policy, value)
    #
    # print('')
    #
    # print('## Q-learning')
    # policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    # env.render(policy, value)
    #
    # print('')
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
    #                                gamma, epsilon, seed=seed)
    # policy, value = linear_env.decode_policy(parameters)
    # linear_env.render(policy, value)


if __name__ == "__main__":
    main()