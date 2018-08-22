import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

#initialize table as 0
Q = np.zeros([env.observation_space.n, env.action_space.n])
#env.observation_space.n means that 16 and action is 4
#.n means the number of the space

#set up learning parameters
lr = 0.85
y = 0.99
num_episodes = 2000

#make list which includes total reward
rList = []
for i in range(num_episodes):
    #reset env and get the first obs
    s = env.reset()
    rAll = 0
    d = False
    j = 0

    # Learning Algorithm for Q table
    while j < 99:
        j += 1
        # Choose action as greedy from Q table being with noise
        a = np.argmax(Q[s, : ] + np.random.randn(1, env.action_space.n) * (1./(i+1)))
        # Return a sample (or samples) from the "standard normal" distribution.
        # 1 by 4 standard normal distribution
        # discount factor??

        #new state and reward from env
        s1, r, d, _ = env.step(a)

        # update Q table through new info
        Q[s, a] = Q[s, a] + lr * ( r + y * np.max(Q[s1, :]) - Q[s, a])
        rAll += r
        s = s1

        if d == True:
            break


    rList.append(rAll)

print(rList)
print("Score over time: " + str(sum(rList) / num_episodes))
print("Final Q-Table values")
print(Q)

plt.plot(rList)
plt.show()






