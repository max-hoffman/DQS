import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import sys

# example data:
#  - 3 functions
#  - 10000 data points
#  - single batch

def generate_data(inputs, outputs):
    data = np.zeros((inputs, 1))
    for i in range(inputs): 
        data[i][0] = random.randint(0,10)

    oracle = np.zeros((1, outputs))
    for i in range(outputs):
        oracle[0][i] = random.randint(0, 5)
    # data_tf = tf.convert_to_tensor(data, np.float32)
    # oracle_tf = tf.convert_to_tensor(oracle, np.float32)
    return data, oracle

DATA_LENGTH = 1000
RNN_HIDDEN    = 50
OUTPUT_SIZE   = 20      # oracle length
NON_ZERO_PENALTY = 1
TINY          = 1e-6    # to avoid NaNs in logs
LEARNING_RATE = 0.01
LOGS_PATH = 'logs'

example_data, example_oracle = generate_data(DATA_LENGTH, OUTPUT_SIZE)

with tf.name_scope('input'):
    inputs = tf.placeholder(tf.float32, shape=(None, None, 1)) # (batch, time, in)
    outputs = tf.placeholder(tf.float32, shape=(None, None, 20))

with tf.name_scope('forward-pass'):
    cell = tf.nn.rnn_cell.LSTMCell(num_units=RNN_HIDDEN)

    batch_size    = tf.shape(inputs)[0]
    initial_state = cell.zero_state(batch_size, tf.float32)

    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)

    # final_projection = lambda x: layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=None)

    # predicted_outputs = tf.map_fn(final_projection, rnn_outputs)
    predicted_outputs = layers.linear(rnn_outputs, num_outputs=OUTPUT_SIZE, activation_fn=None)

with tf.name_scope('error'):
    mean_square_error = tf.reduce_mean(tf.squared_difference(predicted_outputs, outputs))
    mag_error = tf.reduce_mean(predicted_outputs)
    # TODO : fix error for number of non-zero parameters
    # non_zero_error = NON_ZERO_PENALTY * np.count_nonzero(predicted_outputs)
    error_op = mean_square_error + mag_error

with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error_op)

with tf.name_scope('accuracy'):
    # this should be ran with a pre-set training set on a regular basis (ex: every 100)
    accuracy = tf.reduce_mean(tf.cast(tf.abs(outputs - predicted_outputs) < 0.5, tf.float32))

tf.summary.scalar('error', error_op)
summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

# loop
EPOCHS = 100
ITERATIONS_PER_EPOCH = 10

session = tf.Session()
session.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(LOGS_PATH , session.graph)

for epoch in range(EPOCHS):
    epoch_error = 0
    for i in range(ITERATIONS_PER_EPOCH):
        _, error, summary = session.run([train_op, error_op, summary_op], 
            feed_dict={inputs: [example_data], outputs: [example_oracle]})
        writer.add_summary(summary, epoch + i)
        epoch_error += error
    epoch_error /= ITERATIONS_PER_EPOCH
    print("Epoch: %s, error: %.2f" % (epoch, epoch_error))
    guess = session.run(predicted_outputs, feed_dict={ inputs: [example_data] })
    print(guess[0][-1])
writer.close()
    # valid_accuracy = session.run(accuracy, {
    #     inputs:  valid_x,
    #     outputs: valid_y,
    # })
    # print "Epoch %d, train error: %.2f, valid accuracy: %.1f %%" % (epoch, epoch_error, valid_accuracy * 100.0)