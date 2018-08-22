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
        self.conv2 = slim.conv2d(inputs = self.conv1, num_outputs = 64, kernel_size=[4, 4], stride = [2, 2], \
                                 padding='VALID', biases_initializer=None)
        self.conv3 = slim.conv2d(inputs = self.conv2, num_outputs = 64, kernel_size=[3, 3], stride = [1, 1], \
                                 padding='VALID', biases_initializer=None)
        self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1], \
                                 padding='VALID', biases_initializer=None)

        #get output from the last layer and divide




