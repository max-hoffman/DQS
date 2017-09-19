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

def runDQS():
  functionCount = 3
  maxOrder = 3
  maxElements = 3
  coefficientMagnitude = 2
  henkelRows = 10
  dt = .001

  epochs = 1
  epsilon = .9

  for epoch in range(epochs):
    data, xiOracle = generateTrainingSample(functionCount, maxOrder, maxElements, coefficientMagnitude)
    V, dX, theta, norms = initializeSINDy(data[:, 0], henkelRows, functionCount, maxOrder, dt)
    xi, resid, rank, s = np.linalg.lstsq(theta, dX)
    state = theta
    # action space = np.zeros(theta.shape)
    # actions.flatten()
  
    status = 1
    while status:

      # get action estimates
      qVals = predict(state, batchSize=1)

      # select action
      if rd.random > epsilon:
        action, maxQ = randomAction(state)
      else: 
        action, maxQ = greedyAction(qVals)
      
      # take action
      newState = transition(state, action)

      # observe reward
      reward = rewardAt(newState, xiOracle)

      # get targetQ with look-ahead
      newQVals = predict(newState, batchSize=1)
      newAction, lookforwardQ = greedyAction(newQVals)
      targetQ, terminal = checkTarget(action, lookforwardQ, reward)
      qUpdate = qVals
      qUpdate[action] = targetQ

      # update model with target values
      fitModel(state, qUpdate)

def checkTarget(action, newQ, reward):
  terminal = False
  # if reward is >=0, newQ = reward
  # otherwise add the two
  # if all three >=0, terminal
  return newQ, terminal

# state has xi
def transition(state, action):
  return state

# actions : 
# - one per vector
# - can zero out, or set to one any row in vector
def greedyAction(qVals):
  action = []
  for col in qVals:
    new = np.argmax(col)
    action.append(new)
  return action

def randomAction(state):
  action = []
  for i in range(state):
    new = rd.randint(0, len(state[0]))
    if rd.random > .5:
      new = -new
    action.append(new)
    # get Q's, zip before return
  return action

# value net will be random number generator to begin with
# return return Q vals for each column
def predictState(state, batchSize):
  return

# reward function : -1 if continuing, or +10 depending on if state is terminal
def rewardAt(state, oracle):
  return -1

def fitModel(state, qUpdate, batchSize=1, nbEpoch=1, verbose=1):
  return

# experience replay array?