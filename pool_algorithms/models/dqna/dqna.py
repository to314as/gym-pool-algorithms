import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from . import dqna_agent
import torch

def save_model(filepath, model):
    torch.save(model.qnetwork_local.state_dict(), filepath)

def load_model(modelpath, model_params):
    state_size = model_params['s_dim']
    #print('State size is ' + str(state_size))
    action_buckets = model_params['buckets']
    #print('Action buckets are ' + str(action_buckets))
    action_size = action_buckets[0] * action_buckets[1]
    #print('Action size is ' + str(action_size))

    agent = dqna_agent.Agent(state_size, action_size, seed = 229)
    agent.qnetwork_local.load_state_dict(torch.load(modelpath))
    #print(agent.qnetwork_local.state_dict())

    return agent


def action_to_tuple(action, action_buckets):
    #print('Action buckets are ' + str(action_buckets))
    return(float(int(action) % action_buckets[0]),\
        int(action/action_buckets[0]))

def choose_action(state, model, action_space, epsilon = 0.):
    action = action_to_tuple(model.act(state, epsilon), action_space.buckets)
    print('action was ' + str(action))
    return action

def train(env, model_path, episodes=200, episode_length=50):
    learning_rates = [0.1]#[0.15,0.01]#[0.1,0.05,0.15,0.01]
    learning_decay = [0.00001]
    discount_factors = [0.95]#[0.99,0.95,0.9]
    updates =[4]
    performance={}
    #episodes = 1500; episode_length = 25
    for lr in learning_rates:
        for d in learning_decay:
            for gamma in discount_factors:
                for u in updates:
                    dqna_agent.LR = lr; dqna_agent.DECAY = d; dqna_agent.GAMMA = gamma; dqna_agent.UPDATE_EVERY = u
                    results = []; lengths =[]; avg_results = []; avg_lengths = []
                    # Initialize DQN Agent
                    state_size = env.state_space.n
                    action_buckets = [360, 1]
                    env.set_buckets(action=action_buckets)
                    action_size = action_buckets[0] * action_buckets[1]
                    model_params = {'s_dim': env.state_space.n,
                                    'a_dim': env.action_space.n,
                                    'buckets': env.action_space.buckets}
                    print('model params: ', env.state_space.n, env.action_space.n, env.action_space.buckets)
                    agent = dqna_agent.Agent(state_size, action_size, seed = 229)
                    #agent = load_model('saved_model_3balls',model_params)
                    # Learning related constants; factors should be determined by trial-and-error
                    get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/25))) # epsilon-greedy, factor to explore randomly; discounted over time

                    # Q-learning
                    for i_episode in range(episodes):
                        epsilon = get_epsilon(i_episode)

                        state = env.reset() # reset environment to initial state for each episode
                        rewards = 0 # accumulate rewards for each episode
                        done = False
                        for t in range(episode_length):
                            # Agent takes action using epsilon-greedy algorithm, get reward
                            action = agent.act(state, epsilon)
                            next_state, reward, done = env.step(action_to_tuple(action, action_buckets))
                            rewards += reward
                            #print('The reward is ' + str(reward))
                            #print('The next state is ' + str(next_state))

                            # Agent learns over New Step
                            agent.step(state, action, reward, next_state, done)

                            # Transition to next state
                            state = next_state



                            if done:
                                print('Episode {} finished after {} timesteps, total rewards {}'.format(i_episode, t+1, rewards))
                                results.append(rewards)
                                lengths.append(t)
                                avg_lengths.append(np.mean(lengths[-10:]))
                                avg_results.append(np.mean(results[-10:]))
                                with open("output\\dqn-log.txt", "a") as myfile:
                                    myfile.write('Episode {} finished after {} timesteps, total rewards {}\n'.format(i_episode, t+1, rewards))
                                break
                        if not done:
                            print('Episode {} finished after {} timesteps, total rewards {}'.format(i_episode, episode_length, rewards))
                            results.append(rewards)
                            lengths.append(episode_length)
                            avg_lengths.append(np.mean(lengths[-10:]))
                            avg_results.append(np.mean(results[-10:]))
                            with open("output\\dqn-log.txt", "a") as myfile:
                                myfile.write('Episode {} finished after {} timesteps, total rewards {}\n'.format(i_episode, episode_length, rewards))

                        save_model('saved_model_3balls', agent)
                    performance[4,lr, d, gamma, u] = [np.mean(results), np.mean(lengths)]
                    with open("output\\experiments.txt", "a") as myfile:
                        myfile.write('Transfer Learning Balls {} lr{} decay{}, gamma{} updates {} : {}; {}\n'.format(3,lr,d,gamma,u,np.mean(results),np.mean(lengths)))
                    print(results)
                    plt.plot(lengths)
                    plt.plot(avg_lengths)
                    plt.xlabel('Episodes')
                    plt.ylabel('Lengths')
                    plt.title('Transfer Learning Balls {} lr{} decay{}, gamma{} updates{}'.format(3,lr,d,gamma, u))
                    plt.savefig('output\\lengths_3_3.png')
                    plt.show()
                    plt.plot(results)
                    plt.plot(avg_results)
                    plt.xlabel('Episodes')
                    plt.ylabel('rewards')
                    plt.title('Transfer Learning Balls {} lr{} decay{}, gamma{} updates {}'.format(3,lr,d,gamma, u))
                    plt.savefig('output\\rewards_3_3.png')
                    plt.show()

                    print("avg results ",lr, d, gamma, np.mean(results))
                    print("avg length",lr, d ,gamma, np.mean(lengths))