import random as rd
import matplotlib.pyplot as plt
import pylab
import numpy as np
from collections import deque
import tensorflow as tf


# function_count = 3
# max_order = 3
# max_elements = 3
# coefficient_magnitude = 2
# henkel_rows = 10
# dt = .001
# data, xiOracle = generateTrainingSample(function_count, maxOrder, max_elements, coefficientMagnitude)
# V, dX, theta, norms = initializeSINDy(data[:, 0], henkel_rows, function_count, maxOrder, dt)
# xi, resid, rank, s = np.linalg.lstsq(theta, dX)
# print("xi", xi)
# print("resid", resid)

class DQSAgent:
    def __init__(self, state_size, action_size, sesh, logs_path):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.sesh = sesh
        self.training_step = 0

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, shape=[None, state_size])
            self.y_ = tf.placeholder(tf.float32, shape=[None, action_size])

        with tf.name_scope('weights'):
            # model parameters change, so we want tf.Variables
            W = tf.Variable(tf.zeros([state_size, action_size]))
            b = tf.Variable(tf.zeros([action_size]))

        with tf.name_scope('softmax'):
            # attach property for prediction outside of class
            self.y = tf.nn.softmax(tf.matmul(self.x, W) + b)

        with tf.name_scope('squared-error'):
            squared_error = tf.reduce_sum(tf.squared_difference(self.y, self.y_))

        with tf.name_scope('train'):
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(squared_error)

        # track cost and accuracy
        tf.summary.scalar("cost", squared_error)
        self.summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
        self.writer = tf.summary.FileWriter(logs_path , self.sesh.graph)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        return
    
    def train(self, state, target, epoch, dim):
        _, summary = self.sesh.run([self.train_op, self.summary_op], feed_dict={ self.x: [state], self.y_: target })
        self.writer.add_summary(summary, epoch + dim)
        if self.training_step % 50 == 0 & dim == 0:
            print("Epoch, step: ", epoch, self.training_step)
        return

    def _predict(self, state):
        return self.sesh.run(self.y, feed_dict={ self.x: [state] })

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return rd.randrange(self.action_size)
        q = self._predict(state)
        return np.argmax(q)

    def step(self, state, action, oracle, theta, dX):
        # change the state according to the action
        sparsify = True
        next_state = state
        if action == 52:
            return next_state, -1, False
        elif action > self.state_size - 1:
            action = action % self.state_size
            sparsify = False

        next_state[action] = 0 if sparsify else 1
        next_state = self._lsqr(next_state, theta, dX)
        reward, done = self._reward(next_state, oracle)

        return next_state, reward, done
        
    def _lsqr(self, state, theta, dX):
        big_idx = abs(state[:]) > 0
        temp_state, _, _, _ = np.linalg.lstsq(theta[:, big_idx], dX[:])
        state[big_idx] = temp_state
        return state

    def _reward(self, state, oracle):
        states = abs(state[:]) > 0
        oracles = abs(oracle[:]) > 0
        if states == oracles:
            return 0, True
        return -1, False

    def target(self, state, action, reward, done):
        # lookahead Q values
        # TODO: target dimensionality is 1x41 ; is this correct?
        target = self._predict(state)
        target[0][action] = reward if done else reward + target[0][action]
        return target


    # def replay(self, batch_size, i):
    #     minibatch = random.sample(self.memory, batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         target = reward
    #         if not done:
    #             target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
    #         # target_f = self.model.predict(state)
    #         # target_f[0][action] = target
    #         # self.model.fit(state, target_f, epochs=1, verbose=0)
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay
    #     return