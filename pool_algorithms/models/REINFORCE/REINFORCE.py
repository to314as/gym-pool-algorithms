# taken loosely from https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.001)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
       # print('probs', probs)
        if np.random.uniform() >1:
            highest_prob_action = np.random.choice(self.num_actions)
        else:
            highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

    def action_helper_method(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        probs = probs.squeeze(0)
        #print("PROBABILITY", str(probs))
        #for p in probs:
        #    if not p == 0:
        #        print('A VALUE', p)
        log_prob = []
        size = list(probs.size())[0]
        for p in range(size):
            log_prob.append(torch.log(probs.squeeze(0)[p]))
        return probs, log_prob

def update_policy(policy_network, rewards, log_probs, states, p_net,GAMMA = 0.9, scale = 10):
    print('policy update')
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA ** pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards).type('torch.FloatTensor')
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-9)  # normalize discounted rewards

    policy_gradient = []
    #Expected rewards
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        if  not torch.isinf(-log_prob *Gt):
            policy_gradient.append(-log_prob * Gt)

    #entropy bit

    for s in states:
        total = 0
        probs, lprob = p_net.action_helper_method(s)
        for i in range(len(probs)):
            if not probs[i] ==0:
                total += (probs[i]*lprob[i])
        total *= -1
        total *= scale
        if not torch.isinf(total):
            policy_gradient.append(total)

    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()

def action_to_tuple(action, action_buckets):
    angle = np.floor(action/ action_buckets[1])
    power = (action%action_buckets[1])
    return (angle, 0)#power)

def train(env,lr=0.01, decay=0, episodes = 100, episode_length = 25):
    action_buckets = [360, 1]
    env.set_buckets(action=action_buckets)
    print('types: ', env.state_space.n, env.action_space.n, env.action_space.buckets)
    policy_net = PolicyNetwork(env.state_space.n, (env.action_space.buckets[0]*env.action_space.buckets[1]), 128, learning_rate=0.01)

    numsteps = []; avg_numsteps = []; all_rewards = []; avg_rewards = []
    episodes = 100; scale = 50; episode_length = 50
    for episode in range(episodes):

        state = env.reset()
        log_probs = []
        rewards = []
        done = False
        episodes_states = [state]
        for steps in range(episode_length):
            action, log_prob = policy_net.get_action(state)

            new_state, reward, done = env.step(action_to_tuple(action, action_buckets))#action)
            log_probs.append(log_prob)
            rewards.append(reward)
            if done or steps == episode_length-1:
                update_policy(policy_net, rewards, log_probs, episodes_states,policy_net,GAMMA = decay,scale=scale)
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards))
                avg_rewards.append(np.mean(all_rewards[-10:]))
                if episode % 1 == 0:
                    sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode,
                                                                                                              np.round(
                                                                                                                  np.sum(
                                                                                                                      rewards),
                                                                                                                  decimals=3),
                                                                                                              np.round(
                                                                                                                  np.mean(
                                                                                                                      avg_rewards[
                                                                                                                      -10:]),
                                                                                                                  decimals=3),
                                                                                                              steps))
                break
            episodes_states.append(new_state)
            state = new_state
        if done == False:
            numsteps.append(steps)
            avg_numsteps.append(np.mean(numsteps[-10:]))
            all_rewards.append(np.sum(rewards))
            if episode % 1 == 0:
                sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode,
                                                                                                          np.round(
                                                                                                              np.sum(
                                                                                                                  rewards),
                                                                                                              decimals=3),
                                                                                                          np.round(
                                                                                                              np.mean(
                                                                                                                  avg_rewards[
                                                                                                                  -10:]),
                                                                                                              decimals=3),
                                                                                                          steps))

    with open("output\\experiments.txt", "a") as myfile:
        myfile.write('REINFORCE Balls {} lr{} decay{} scale {} : {}; {}\n'.format(2, lr, decay,scale,np.mean(all_rewards),np.mean(numsteps)))

    plt.plot(numsteps)
    plt.plot(avg_numsteps)
    plt.xlabel('Episodes')
    plt.ylabel('Lengths')
    plt.title('REINFORCE Balls{} lr{} decay{}'.format(2, lr, decay))
    plt.savefig('output\\lengths_3_3.png')
    plt.show()
    plt.plot(all_rewards)
    plt.plot(avg_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('rewards')
    plt.title('TREINFORCE Balls {} lr{} decay{}'.format(2, lr, decay))
    plt.savefig('output\\rewards_3_3.png')
    plt.show()

    print("avg results ", lr, decay, np.mean(all_rewards))
    print("avg length", lr, decay, np.mean(numsteps))
    return sum(numsteps)/len(numsteps), sum(all_rewards)/len(all_rewards), len(numsteps)

