import random as rd
import matplotlib.pyplot as plt
import pylab
import numpy as np
import tensorflow as tf
from initializeSINDy import *
from sampleGenerator import generateTrainingSample

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
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self._build_model()
    
    def _build_model(self):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, shape=[None, state_size])
            y_ = tf.placeholder(tf.float32, shape=[None, action_size])

        with tf.name_scope('weights'):
            # model parameters change, so we want tf.Variables
            W = tf.Variable(tf.zeros([26, 53]))
            b = tf.Variable(tf.zeros([53]))

        with tf.name_scope('softmax'):
            # prediction
            y = tf.nn.softmax(tf.matmul(x, W) + b)

        with tf.name_scope('cross-entropy'):
            squared_error = tf.reduce_sum(tf.squared_difference(y, y_))

        with tf.name_scope('train'):
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(squared_error)

        # track cost and accuracy
        tf.summary.scalar("cost", squared_error)

        # merge summaries for convenience
        self.summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        return
    
    def train(self, sesh, state, target, epoch, i):
        # train the model given the current state, and the target Q function
        # void return
        _, summary = sesh.run([self.train_op, self.summary_op], feed_dict={ x: state, y_: target })
        writer.add_summary(summary, epoch + i)
        if epoch % 5 == 0 & i == 0:
            print("Epoch: ", epoch)
        return

    def action(self, state, i):
        # get the value function(53x1) for state i (26x1)
        # find argmax
        # decide whether to take greedy action or not
        # return the action
        return

    def step(self, action, i):
        # change the state according to the action
        # lsq optimize the state again
        # calculate reward based on whether done or not
        # return next_state, reward, done
        return

    def target(self, state, action, reward, i):
        # lookahead Q values
        # find max Q in lookahead

        return

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


if __name__ == "__main__":
    with tf.Session() as sesh:
        state_size = 26
        action_Size = 53
        function_count = 3
        max_order = 3
        max_elements = 3
        coefficient_magnitude = 2
        henkel_rows = 10
        dt = .001
        terminal = False
        logs_path = '../logs'

        agent = DQSAgent(state_size, action_size)
        sesh.run(tf.initialize_all_variables())
        writer = tf.summary.FileWriter(logs_path , sesh.graph)

        for epoch in range(training_epochs):
            data, xiOracle = generateTrainingS_cmple(function_count, max_order, max_elements, coefficient_magnitude)
            V, dX, theta, norms = initializeSINDy(data[:, 0], henkel_rows, function_count, max_order, dt)
            xi, resid, rank, s = np.linalg.lstsq(theta, dX)
            
            for i in range(len(xi[0])):

                # take next action
                action = agent.action(state, i)

                # get reward (at new state)
                next_state, reward, done = agent.step(action, i)
                if done:
                    continue

                # get target value
                target = agent.target(next_state, action, reward, i)
                agent.remember(state, action, reward, next_state, done)

                # train model
                agent.train(sesh, state, target, epoch, i)

                # experience replay
                # if len(agent.memory) > batch_size:
                #     agent.replay(batch_size, i)
                
                # update state for next round
                state = next_state  

        print("Accuracy: ", accuracy.eval(feed_dict={ x: mnist.test.images, y_: mnist.test.labels }))
        print("done")

# def trainDQS(epochs=1, epsilon=.9):
#   function_count = 3
#   max_order = 3
#   max_elements = 3
#   coefficient_magnitude = 2
#   henkel_rows = 10
#   dt = .001

#   for epoch in range(epochs):
#     data, xiOracle = generateTrainingSample(function_count, max_order, max_elements, coefficient_magnitude)
#     V, dX, theta, norms = initializeSINDy(data[:, 0], henkel_rows, function_count, max_order, dt)
#     xi, resid, rank, s = np.linalg.lstsq(theta, dX)
    
#     # initialize RL vars
#     state = np.append(resid, xi)  # xi and residuals
#     action = np.zeros(state.shape) # selecting resid does nothing
#     status = 1
#     cumulativeReward = 0
#     rewards = []

#     while status:

#       # get action estimates
#       qVals = predict(state, batchSize=1)

#       # select action
#       if rd.random > epsilon:
#         action, expectedQ = randomAction(state)
#       else: 
#         action, expectedQ = greedyAction(qVals)
      
#       # take action
#       newState = transition(state, action)

#       # observe reward
#       reward = rewardAt(newState, xiOracle)
#       cumulativeReward += reduce((lambda x, y: x + y), reward)

#       # get target Q-values with look-ahead
#       newQVals = predict(newState, batchSize=1)
#       newAction, lookaheadQ = greedyAction(newQVals)
#       updates, terminal = getTarget(lookaheadQ, reward)
#       targetQ = qVals
#       targetQ[action, :] = updates

#       # update model with target values
#       fitModel(state, targetQ)

#       # next steps
#       state = newState
#       status = terminal
#     rewards.append(cumulativeReward)
#   return rewards

# # get the target value given the lookahead and reward
# def getTarget(loohahead, reward):
#   terminalCount = 0
#   target = loohahead + reward
#   for i in range(len(current)):
#     if reward[i] == 0:
#       target[i] = 0
#       terminalCount += 1
#   terminal = False
#   if terminalCount == 0:
#     terminal = True
#   return terminal, target

# # change state, regress new state, return new state
# def transition(state, action):
#   return state

# # find the max Q val action in each column
# def greedyAction(qVals):
#   action = []
#   for col in qVals:
#     new = np.argmax(col)
#     action.append(new)
#   return action

# # return a random action vector
# def randomAction(state):
#   action = []
#   for i in range(state):
#     new = rd.randint(0, len(state[0]))
#     if rd.random > .5:
#       new = -new
#     action.append(new)
#     # get Q's, zip before return
#   return action

# # return Q vals for each possible action
#   # take action, optimize, estimate value of new state
# def predict(state, batchSize):
#   return

# # reward is 0 if oracle matched, -1 otherwise
# def rewardAt(state, oracle):
#   return -1

# # backpropogation on Q function
# def fitModel(state, targetQ, batchSize=1, nbEpoch=1, verbose=1):
#   return

# # experience replay array?
