import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from displayfxn import showOperation

env = gym.make('FrozenLake-v0')

tf.reset_default_graph()

#feedforward part of network used for choosing action
inputs1 = tf.placeholder(shape = [1, 16], dtype = tf.float32)
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
#tf.random_uniform makes 16 by 4 tensor matrix with random number. minimum value is 0 and maximum is 0.01
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1) #at axis = 1 (column), return maximum value

#(Q_target - Q_out_current)^2 = loss
nextQ = tf.placeholder(shape=[1, 4], dtype = tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate= 0.1) #GDS
updateModel = trainer.minimize(loss)

#Network learning
init = tf.global_variables_initializer()

#set up learning parameter
y = 0.99
e = 0.2
num_episodes = 2500

#generate lists taking total rewards and step at each episode

jList = []
rList = []

#be
with tf.Session() as sess:
    sess.run(init)

    for i in range(num_episodes):
        #reset env and get the first observation
        s = env.reset()

        rAll = 0
        d = False
        j = 0

        #Q network
        while j < 99:
            j += 1

            # Choose action with greedy from Q network adding random action with e possibility
            a, allQ = sess.run([predict, Qout], feed_dict={inputs1:np.identity(16)[s:s+1]})
            #np.identity(16): makes 16 by 16 identical matrix(1, 0, ...; 0, 1......)
            #if s = 0 -> [1, 0, 0, ....] / if s = 1 -> [0, 1, 0, ...] -> one hot encoding for state(obs)
            #a size is 1 by 4
            if np.random.rand(1) < e: #e = 0.1 -> 10% random, 90% learning
                a[0] = env.action_space.sample()
            #a[0]?? pull out [ .. ] from [[ ... ]]
            #get new status and reward from env
            s1, r, d, _ = env.step(a[0]) #action is input

            #get Q value feeding new status s1 to network

            Q1 = sess.run(Qout, feed_dict={inputs1:np.identity(16)[s:s+1]}) # 1 by 4
            #predict is chosen action, Qout is action pool?

            #get maxQ and set up target value about selected action a
            maxQ1 = np.max(Q1) # choose max value from 1 by 4 matrix of Q1

            targetQ = allQ # 1 by 4
            targetQ[0, a[0]] = r + y * maxQ1 # change the value of action
            #a[0] is 0, 1, 2, or 3
            #above line revised the value of Q in the 1 by 4 array of Q with specific action a[0]

            #?????????

            # by using targeted and predicted Q value, learn network
            _, W1 = sess.run([updateModel, W], feed_dict={inputs1: np.identity(16)[s:s+1], nextQ: targetQ})

            rAll += r
            s = s1
            if d == True:
                # as learned model, the possibility to choose random action is reduced
                e = 1./((i / 50 ) + 10)
                break
        jList.append(j)
        rList.append(rAll)

print("Percent of successful episodes: " + str(sum(rList)/num_episodes))

plt.plot(rList)
#plt.plot(jList)
plt.show()










