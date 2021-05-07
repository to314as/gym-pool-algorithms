import argparse
import sys

from .q_table import q_table
from .q_table import q_table_new
from .dqn import dqn
from .dqna import dqna
from .REINFORCE import REINFORCE
import gym
import gym_pool

env=gym.make('gym_pool:Pool-v0')

EPISODES = 1000
EPISODE_LENGTH = 25

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL training.')
    parser.add_argument('--output_model', type=str, default='q_table_new_output_model',
            help='Output model path.')
    parser.add_argument('--algo', type=str, default='q-table',
            help='One of q-table, dqn (Deep Q-Network) or REINFORCE. Default: q-table')
    parser.add_argument('--balls', type=int, default=2,
            help='Number of balls on table (including white ball), should be >= 2. Default: 2')
    parser.add_argument('--visualize', dest='visualize', action='store_true',
            help='To see the visualization of the pool game.')
    args = parser.parse_args()

    if args.balls < 2:
        print('Number of balls should be >= 2.')
        sys.exit(1)

    single_env = True
    
    if args.algo == 'q-table':
        env = gym.make('gym_pool:Pool-v1')
        env.visualize=args.visualize
        #algo = q_table.train
        algo = q_table_new.train
    elif args.algo == 'dqn':
        env = gym.make('gym_pool:Pool_continuous-v0')
        env.visualize = args.visualize
        algo = dqn.train
    elif args.algo == 'dqna':
        env = gym.make('gym_pool:Pool_angle-v0')
        env.visualize = args.visualize
        algo = dqna.train
    elif args.algo == 'REINFORCE':
        env.visualize = args.visualize
        algo = REINFORCE.train
    else:
        print('Algorithm not supported! Should be one of q-table, dqn, dqna or REINFORCE.')
        sys.exit(1)

    if single_env:
        env = env
        algo(env, args.output_model, episodes=EPISODES, episode_length=EPISODE_LENGTH)
    else:
        env_params = { 'num_balls': args.balls, 'visualize': args.visualize }
        algo(env_params, args.output_model, episodes=EPISODES, episode_length=EPISODE_LENGTH)
