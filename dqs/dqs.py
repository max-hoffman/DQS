from initializeSINDy import *
from sampleGenerator import generateTrainingSample
import matplotlib.pyplot as plt
import pylab
import numpy as np
import random as rd

# functionCount = 3
# maxOrder = 3
# maxElements = 3
# coefficientMagnitude = 2
# henkelRows = 10
# dt = .001
# data, xiOracle = generateTrainingSample(functionCount, maxOrder, maxElements, coefficientMagnitude)
# V, dX, theta, norms = initializeSINDy(data[:, 0], henkelRows, functionCount, maxOrder, dt)
# xi, resid, rank, s = np.linalg.lstsq(theta, dX)
# print("xi", xi)
# print("resid", resid)

def trainDQS(epochs=1, epsilon=.9):
  functionCount = 3
  maxOrder = 3
  maxElements = 3
  coefficientMagnitude = 2
  henkelRows = 10
  dt = .001

  for epoch in range(epochs):
    data, xiOracle = generateTrainingSample(functionCount, maxOrder, maxElements, coefficientMagnitude)
    V, dX, theta, norms = initializeSINDy(data[:, 0], henkelRows, functionCount, maxOrder, dt)
    xi, resid, rank, s = np.linalg.lstsq(theta, dX)
    
    # initialize RL vars
    state = np.append(resid, xi)  # xi and residuals
    action = np.zeros(state.shape) # selecting resid does nothing
    status = 1
    cumulativeReward = 0
    rewards = []

    while status:

      # get action estimates
      qVals = predict(state, batchSize=1)

      # select action
      if rd.random > epsilon:
        action, currentQ = randomAction(state)
      else: 
        action, currentQ = greedyAction(qVals)
      
      # take action
      newState = transition(state, action)

      # observe reward
      reward = rewardAt(newState, xiOracle)
      cumulativeReward += reduce((lambda x, y: x + y), reward)

      # get target Q-values with look-ahead
      newQVals = predict(newState, batchSize=1)
      newAction, lookaheadQ = greedyAction(newQVals)
      updates, terminal = getTarget(lookaheadQ, reward)
      targetQ = qVals
      targetQ[action, :] = updates

      # update model with target values
      fitModel(state, targetQ)

      # next steps
      state = newState
      status = terminal
    rewards.append(cumulativeReward)
  return rewards

# get the target value given the lookahead and reward
def getTarget(loohahead, reward):
  terminalCount = 0
  target = loohahead + reward
  for i in range(len(current)):
    if reward[i] == 0:
      target[i] = 0
      terminalCount += 1
  terminal = False
  if terminalCount == 0:
    terminal = True
  return terminal, target

# change state, regress new state, return new state
def transition(state, action):
  return state

# find the max Q val action in each column
def greedyAction(qVals):
  action = []
  for col in qVals:
    new = np.argmax(col)
    action.append(new)
  return action

# return a random action vector
def randomAction(state):
  action = []
  for i in range(state):
    new = rd.randint(0, len(state[0]))
    if rd.random > .5:
      new = -new
    action.append(new)
    # get Q's, zip before return
  return action

# return Q vals for each possible action
  # take action, optimize, estimate value of new state
def predict(state, batchSize):
  return

# reward is 0 if oracle matched, -1 otherwise
def rewardAt(state, oracle):
  return -1

# backpropogation on Q function
def fitModel(state, targetQ, batchSize=1, nbEpoch=1, verbose=1):
  return

# experience replay array?