import math
import pickle
import matplotlib.pyplot as plt

import numpy as np


def choose_action(state, q_table, action_space, epsilon=0.1):
    if np.random.random_sample() < epsilon: # random action
        return action_space.sample()
    else: # greedy action based on Q table
        return [np.argmax(q_table[i][state]) for i in range(len(q_table))]

def save_model(filepath, model):
    with open(filepath, 'wb') as fout:
        pickle.dump(model, fout)

def load_model(filepath):
    with open(filepath, 'rb') as fin:
        model = pickle.load(fin)
    return model

def train(env, model_path, episodes=200, episode_length=50):
    print('Q-Table training')

    env.set_buckets(action=[18, 5], state=[50, 50])
    m = env.num_balls

    # Preparing Q table of size n_states x n_actions
    n_states = env.state_space.n
    n_actions = env.action_space.n
    q_table = [np.zeros(n_states+(n_a,)) for n_a in n_actions] # q_table[i][s] = ai
    print("Q-table shape:",q_table[0].shape)
    #print(len(q_table))

    # Learning related constants; factors should be determined by trial-and-error
    get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/25))) # epsilon-greedy, factor to explore randomly; discounted over time
    get_lr = lambda i: max(0.01, min(0.5, 1.0 - math.log10((i+1)/25))) # learning rate; discounted over time
    gamma = 0.8 # reward discount factor

    # Q-learning
    results = [];
    lengths = [];
    avg_results = [];
    avg_lengths = []
    for i_episode in range(episodes):
        epsilon = get_epsilon(i_episode)
        lr = get_lr(i_episode)

        state = env.reset() # reset environment to initial state for each episode
        rewards = 0 # accumulate rewards for each episode
        done = False
        for t in range(episode_length):
            # Agent takes action using epsilon-greedy algorithm, get reward
            action = choose_action(state, q_table, env.action_space, epsilon)
            next_state, reward, done = env.step(action)
            rewards += reward

            # Agent learns via Q-learning
            for i in range(len(q_table)):
                q_next_max = np.max(q_table[i][next_state])
                q_table[i][state+(action[i],)] += lr * (reward + gamma * q_next_max - q_table[i][state+(action[i],)])

            # Transition to next state
            state = next_state

            if done:
                print('Episode {} finished after {} timesteps, total rewards {}'.format(i_episode, t + 1, rewards))
                results.append(rewards)
                lengths.append(t)
                avg_lengths.append(np.mean(lengths[-10:]))
                avg_results.append(np.mean(results[-10:]))
                break
        if not done:
            print('Episode {} finished after {} timesteps, total rewards {}'.format(i_episode, episode_length, rewards))
            results.append(rewards)
            lengths.append(episode_length)
            avg_lengths.append(np.mean(lengths[-10:]))
            avg_results.append(np.mean(results[-10:]))

    save_model(model_path, q_table)
    print(results)
    plt.plot(lengths)
    plt.plot(avg_lengths)
    plt.xlabel('Episodes')
    plt.ylabel('Lengths')
    plt.title('Q Table Performance')
    plt.savefig('output\\qtable_lengths_3_3.png')
    plt.show()
    plt.plot(results)
    plt.plot(avg_results)
    plt.xlabel('Episodes')
    plt.ylabel('rewards')
    plt.title('Q Table Performance')
    plt.savefig('output\\qtable_rewards_3_3.png')
    plt.show()