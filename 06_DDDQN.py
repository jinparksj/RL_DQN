#Double DQN : Q_target = r + y * Q (s1, argmax(Q(s1, a, theta), theta1)
#Dueling DQN : Q can be devided as value function V(s), which is how good the state is numerically and A(a)
#, which is advantage function, which how much chosen action is better than othe actions
#Q(s, a) = V(s) + A(a)

#1. load library
import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os

from gridworld import gameEnv

env = gameEnv(partial = False, size = 5)

#2. Make network
class Qnetwork():
    def __init__(self, h_size):
        #network get a frame from game and make the frame as array (flattening)
        #and then, reshape the array, put it to four convolution layers as process to build up network, policy

        self.scalarInput = tf.placeholder(shape = [None, 21168], dtype = tf.float32)
        # 21168 = 84 pixel * 84 pixel * 3 (RGB)

        self.imageIn = tf.reshape(self.scalarInput, shape = [-1, 84, 84, 3])
        # make scalarInput (-1 by 21168) to (-1, 84, 84, 3) tensor
        self.conv1 = slim.conv2d(inputs = self.imageIn, num_outputs = 32, kernel_size=[8, 8], stride = [4, 4], \
                                 padding='VALID', biases_initializer=None)
        #8 x 8 kernel, 4 strides, output size is ( 84 - 8 ) / 4 + 1 = 20
        #ouput size of conv layer: ( input size - kernel size ) / # of strides + 1 at no zero padding 'VALID'
        #therefore, 20 x 20 x 32 size of activation output
        self.conv2 = slim.conv2d(inputs = self.conv1, num_outputs = 64, kernel_size=[4, 4], stride = [2, 2], \
                                 padding='VALID', biases_initializer=None)
        #4 x 4 kernel, 2 strides, ( 20 - 4 ) / 2 + 1 = 9,
        #9 x 9 x 64
        self.conv3 = slim.conv2d(inputs = self.conv2, num_outputs = 64, kernel_size=[3, 3], stride = [1, 1], \
                                 padding='VALID', biases_initializer=None)
        #(9 - 3) / 1 + 1 = 7, output is 7 x 7 x 64

        self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1], \
                                 padding='VALID', biases_initializer=None)
        #(7 - 7 ) / 1 + 1 = 1, output is 1 x 1 x h_size(512)


        # by using 1 x 1 x 512 ouput from conv4, divide 2
        #get output (h_size) from the last layer, conv4, and divide advantage stream and value stream
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        # split(value, num_or_size_splits, axis=0, num=None, name='split')
        # Splits a tensor into sub tensors.
        #DIVIDE two along axis = 3 and return streamAC and streamVC
        #NEED TO KNOW the size of conv4!!!
        # 1 x 1 x 512 = 1 x 1 x 256 for A and 1 x 1 x 256 for V
        #make 1 x 1 x 512 as vector
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)


        #???????????????
        self.AW = tf.Variable(tf.random_normal([h_size//2, env.actions])) #h_size//2 return integer ignoring under point
        #tf.random_normal returns tensor of h_szie//2 by env.actions size -> 256 x action space
        self.VW = tf.Variable(tf.random_normal([h_size//2, 1]))
        # 256 x 1 for Value function weight

        self.Advantage = tf.matmul(self.streamA, self.AW) #1 x 1 x 256 * 256 x action space size : Advantage
        self.Value = tf.matmul(self.streamV, self.VW) #1 x 1 x 256 * 256 x 1 = 1 x 1 x 1 : Value fuction

        # compose Advantage stream and Value stream in order to get final Q value
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis = 1, keep_dims=True))
        #Value + (Advantage(certain action advantage) - Advantage average(other action adavantage))
        self.predict = tf.argmax(self.Qout, 1) # choose one action from above Qout

        #Loss function: taking square sum of difference between targeted Q and predicted Q
        #target Q
        self.targetQ = tf.placeholder(shape = [None], dtype = tf.float32)

        #predict Q: action!!!!!
        self.actions = tf.placeholder(shape=[None], dtype = tf.int32)

        #one hot encoding of actions holder
        self.actions_oneshot = tf.one_hot(self.actions, env.actions, dtype = tf.float32)
        """
        one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None)
        Returns a one-hot tensor.

        indices = [0, 1, 2]
        depth = 3
        tf.one_hot(indices, depth)  # output: [3 x 3]
        # [[1., 0., 0.],
        #  [0., 1., 0.],
        #  [0., 0., 1.]]

        indices = [[0, 2], [1, -1]]
        depth = 3
        tf.one_hot(indices, depth,
                   on_value=1.0, off_value=0.0,
                   axis=-1)  # output: [2 x 2 x 3]
        # [[[1.0, 0.0, 0.0],   # one_hot(0)
        #   [0.0, 0.0, 1.0]],  # one_hot(2)
        #  [[0.0, 1.0, 0.0],   # one_hot(1)
        #   [0.0, 0.0, 0.0]]]  # one_hot(-1)

        """
        #Get Q from each network action, why multiply and sum???
        # since tensor(tf) has no index, by using them, we can extract one action without index
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_oneshot), axis = 1)
        #Q = sum(Qout * action) -> action
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

#3. Experience replay buffer
#Experience replay: Save experiences in buffer and sample experiences and then provide sampled experiences randomly
class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    # when add experience to buffer, if buffer is full, delete old experience and put new experience
    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience) #list.extend: add the data at the end of list, at the end of buffer, add experience

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5]) #the number of data is 5 saving in replaybuffer

#game frame resize function
def processState(states):
    return np.reshape(states, [21168])

#4. Target Q update to Main Network
#target network's parameter is updated with the main network's parameter
def updateTargetGraph(tfVars, tau):
    #tfVars: trainable variables
    # tau: the ratio of providing target Q from target Q network to learning network
    total_vars = len(tfVars)
    #op_holder: operater saving list
    op_holder = []

    for idx, var in enumerate(tfVars[0: total_vars//2]):
        # half of trainable variables go to main Q network and half go to target network
        op_holder.append(tfVars[idx + total_vars // 2].assign((var.value() * tau) + ((1-tau) * tfVars[idx + total_vars // 2].value())))
        # the first half * tau -> main network * weight
        # the second half * (1 - tau) -> target network * weight
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


#5. network learning
#learning parameters
batch_size = 32 # the number of experience batch used for each learning stage
update_freq = 4 # update frequency at each learning stage
y = .99 # discounted factor for target Q
startE = 1 # start possibility of random action
endE = 0.1 # end possibility of random action
annealing_steps = 10000 # required reducing step for learning from startE to endE
num_episodes = 10000 # the number of game environment episode to train network
pre_train_steps = 10000 # before training, how many random action will be operated
max_epLength = 50 # maximum step of each episode
load_model = False # True or False selection of load saving model
path = "./dqn" #directory of saving model
h_size = 512 #hidden layer, last convolution layer, conv4, size is 512, before dividing as advantage and value functions
tau = 0.001 # the ratio of updating target Q network to main network

#learning process
tf.reset_default_graph()
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

init = tf.global_variables_initializer()
saver = tf.train.Saver() # saving model??

trainables = tf.trainable_variables() # can extract trainable variables

targetOps = updateTargetGraph(trainables, tau) #make values for updating target network

# directory of saving experience
myBuffer = experience_buffer()

#set up possibility of random action
e = startE
# reduce e gradually
stepDrop = (startE - endE) / annealing_steps

#generate list having total reward and step at each episode
jList =[]
rList = []
total_steps = 0

# make a place for saving model
if not os.path.exists(path):
    os.makedirs(path)

# open TF session, launch tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint.path)
    #make target network same with main network
    updateTarget(targetOps, sess)
    #start episodes
    for i in range(num_episodes):
        #initialize experience replay buffer
        episodeBuffer = experience_buffer()
        #initialize env and initial state
        s = env.reset()
        s = processState(s)
        d = False
        rAll = 0
        j = 0

        # Q Network
        # terminate agent when 50 steps
        while j < max_epLength:
            j += 1

            # action 1. greedy action from Q network or 2. random action with possibility e
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, 4)
            else:
                # get Q value through neural network
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]

            s1, r, d = env.step(a)
            s1 = processState(s1)
            total_steps += 1

            #save experience to episode buffer, current state, action, reward, next state, and done of termination of episode
            episodeBuffer.add(np.reshape([s, a, r, s1, d], [1, 5]))

            # start, when total steps are more than pretraining random action
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop

                # start with update frequency
                if total_steps % (update_freq) == 0:
                    # get randomly batch from experience replay buffer
                    trainBatch = myBuffer.sample(batch_size) # get 32 samples from the buffer, 50000 samples

                    # DDDQN update for target Q
                    # choose action from main Q network s1?????
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:, 3])})

                    # choose Q from target Q network
                    Q2 = sess.run(targetQN.Qout, feed_dict= { targetQN.scalarInput: np.vstack(trainBatch[:, 3])})

                    # make fake index as termination done of True and False
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    #bring Q values from target Q network, the Q values are chosen by Q1, main Q network
                    doubleQ = Q2[range(batch_size), Q1]
                    #reward + discounted factor * doubleQ, targetQ is sum of immediate reward and argmax reward at next state
                    targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)

                    #network update by target value
                    #find loss by difference between target Q and actions
                    _ = sess.run(mainQN.updateModel,\
                                 feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:, 0]), \
                                            mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:, 1]})


                    # set up target Q network as same as main Q network
                    updateTarget(targetOps, sess)

            rAll += r
            s = s1

            if d == True:
                break

        #save buffer of the episode in my buffer
        myBuffer.add(episodeBuffer.buffer)
        #save step
        jList.append(j)
        rList.append(rAll)

        #save model periodically
        if i % 1000 == 0:
            saver.save(sess, path+ '/model-' + str(i) + '.cptk')
            print("Saved Model")

        #reward
        if len(rList) % 10 == 0:
            print(total_steps, np.mean(rList[-10:]), e)

    #save model
    saver.save(sess, path+'/model-' + str(i) + 'cptk')

#success rate
print("Percentage of successful episodes: " + str(sum(rList) / num_episodes))

rMat = np.resize(np.array(rList), [len(rList) // 100, 100])
rMean = np.average(rMat, 1)
plt.plot(rMean)
plt.show()








