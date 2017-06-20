'''
Created on Mar 13, 2017

@author: lujiang
'''


import tensorflow as tf
import numpy as np


def last_relevant(output, length):
  batch_size = tf.shape(output)[0]
  max_length = tf.shape(output)[1]
  out_size = int(output.get_shape()[2])
  index = tf.range(0, batch_size) * max_length + (length - 1)
  flat = tf.reshape(output, [-1, out_size])
  relevant = tf.gather(flat, index)
  return relevant
  
  
tf.reset_default_graph()
tf.set_random_seed(0)

embedding_size = 1
steps = 12
np.random.seed(0)
# Create input data

X = np.ones((3, steps, embedding_size)).astype(np.float32)
X_lengths = [4, 6, 6]
X[1][6:steps] = 0
X[2][9:steps] = 0

cell = tf.nn.rnn_cell.LSTMCell(num_units=embedding_size, state_is_tuple=True)
inX = [tf.squeeze(t, [1]) for t in tf.split(1, steps, X)]
#outputs, last_states = tf.nn.rnn(cell=cell, inputs=inX, sequence_length = X_lengths, initial_state=cell.zero_state(3, tf.float32))
outputs, last_states = tf.nn.rnn(cell=cell, inputs=inX, initial_state=cell.zero_state(3, tf.float32))
print outputs

with tf.Session() as session:
  session.run(tf.initialize_all_variables())
  val1 = session.run(outputs[11], feed_dict=None)
  print val1