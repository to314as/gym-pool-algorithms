""" Brute force implementation for Cartpole """
#! based on the very elegant idea found on https://github.com/kevchn/qlearn-cartpole !#
import gym
import numpy as np
import matplotlib.pyplot as plt
#env = gym.make('gym_pool:Pool_continuous-v0')
#env.visualize=True
class BruteForce:
    def __init__(self):
        self.iter=100
        self.best_reward = 0
        self.durations=[]
        self.rewards = []
        self.best_policy=None

    def run_episode(self,params,params2,env):
        total_reward = 0
        observation = env.reset()
        c=0
        while True:
            angle = params#(params @ observation)/len(observation)
            force = params2#params2 @ observation
            action=(angle, force)
            print(action)
            observation, reward, done = env.step(action)
            print(reward, observation)
            total_reward += reward
            c+=1
            if done or c>25:
                break
        return total_reward,c

    def shot(self,angle, force,env):
        action=(angle, force)
        observation, reward, done = env.step(action)
        return observation, reward, done

    def experiment(self,divisions,df):
        logs=[]
        angles_diff=np.arange(-0.5,0.5,0.5/divisions)
        print(angles_diff)
        force=np.arange(1.,0.,-1./df)[:]
        print(force)
        env = gym.make('gym_pool:Pool_continuous-v0')
        #self.durations = []
        positions=[]
        for i in angles_diff:
            for j in force:
                del env.gamestate
                del env.state_space
                del env.action_space
                del env
                env = gym.make('gym_pool:Pool_continuous-v0')
                env.num_balls=2
                obs = env.current_state
                #print(env.current_state[1])
                observation, reward, done=self.shot(env.current_state[1]+i,j,env)
                #print(observation)
                logs.append([obs,observation,reward,i,j])
                positions.append(observation[0])
                if reward>10:
                    print(i,j)
                    print("reward:",reward)
                    #env.reset()
        env.close()
        print(logs)
        logs = np.array(logs)
        np.save("positions.npy", positions)
        #np.save("brute_force_logs.npy", logs)
        env.close()
        return logs

if __name__=="__main__":
    BF=BruteForce()
    #BF.learn()
    firsts=[]
    means=[]
    #for i in range(5):
    divisions=360
    df=4
    logs=BF.experiment(divisions,df)
    l=np.array(logs)
    reward_surface=l[:,2].reshape(divisions*2,df).astype(np.float64)
    #print(l[:,3].reshape(divisions*2,df))
    #print(l[:,4].reshape(divisions * 2, df))
    fig,ax=plt.subplots(figsize=(15,10))
    plt.imshow(reward_surface,vmin=np.min(reward_surface),vmax=np.max(reward_surface),extent=[0,1,-0.5,0.5])
    plt.colorbar()
    plt.show()
    #means.append(mean)
    #print(np.mean(firsts),np.std(firsts),np.mean(means),np.std(means))