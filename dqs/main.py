import random as rd
import matplotlib.pyplot as plt
import pylab
import numpy as np
import tensorflow as tf

from initializeSINDy import initializeSINDy
from sampleGenerator import generateTrainingSample
from dqs import DQSAgent


if __name__ == "__main__":
      with tf.Session() as sesh:
          function_count = 3
          max_order = 3
          max_elements = 3
          coefficient_magnitude = 2
          henkel_rows = 10
          dt = .001

          state_size = 20
          action_size = 41
          training_epochs = 5
          terminal = False
          logs_path = '../logs'
          epochReward = 0

          agent = DQSAgent(state_size, action_size, sesh, logs_path)
          sesh.run(tf.initialize_all_variables())
          summary = tf.Summary(value=[tf.Summary.Value(tag="reward", simple_value=epochReward)])

          for epoch in range(training_epochs):
              data, oracle = generateTrainingSample(function_count, max_order, max_elements, coefficient_magnitude)
              # TODO: fix that Nan/ Inf problem
              V, dX, theta, norms = initializeSINDy(data[:, 0], henkel_rows, function_count, max_order, dt)
              state, resid, rank, s = np.linalg.lstsq(theta, dX)
              epochReward = 0

              for dim in range(len(state[0])):

                  # take next action
                  action = agent.action(state[:, dim])

                  # get reward (at new state)
                  next_state, reward, done = agent.step(state[:, dim], action, oracle, theta, dX)
                  epochReward += reward
                  if done:
                      continue

                  # get target value
                  target = agent.target(next_state, action, reward, done)
                  agent.remember(state[:, dim], action, reward, next_state, done)

                  # train model
                  agent.train(state[:, dim], target, epoch, dim)

                  # experience replay
                  # if len(agent.memory) > batch_size:
                  #     agent.replay(batch_size, i)
                  
                  # update state for next round
                  state[:, dim] = next_state  

              # record reward at epoch end
              agent.writer.add_summary(summary, global_step=epoch)

      # print("Accuracy: ", accuracy.eval(feed_dict={ x: mnist.test.images, y_: mnist.test.labels }))
      print("done")
