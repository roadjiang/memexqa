'''
Created on Mar 10, 2017

@author: lujiang
'''

import tensorflow as tf
import pickle


def build_lr_embedding_q(placeholders, photo_feat_file, vocabulary_size, num_classes, embedding_size = 300):
  """logistic regression with embedding input"""
  with tf.name_scope('lr_Q'):
    embedding = tf.get_variable("embedding", [vocabulary_size, embedding_size])
    inputs = tf.nn.embedding_lookup(embedding, placeholders["Qs"])
    inputs = tf.reduce_max(inputs, 1)
    # add a fully connected layer
    fc1_w = tf.get_variable("fc1_w", [embedding_size, num_classes], initializer=tf.random_normal_initializer())
    fc1_b = tf.get_variable("fc1_b", [num_classes], initializer=tf.random_normal_initializer())
    fc1_o = tf.nn.sigmoid(tf.matmul(inputs, fc1_w) + fc1_b)
  return fc1_o

def train_lr_embedding_q(loss, global_step):
  return train(loss, global_step, 0.5, optimizer = "GradientDescentOptimizer", decay_steps = 500, max_grad_norm = 5)

def build_lr_embedding_q_i(placeholders, photo_feat_file, vocabulary_size, num_classes, embedding_size = 300):
  """logistic regression with embedding input"""
  with tf.name_scope('lr_Q_I'):
    q_embedding = tf.get_variable("embedding", [vocabulary_size, embedding_size])
    q_vec = tf.nn.embedding_lookup(q_embedding, placeholders["Qs"])
    q_vec = tf.reduce_max(q_vec, 1)
    
    _, photo_feat = pickle.load(open(photo_feat_file, "rb"))
    i_embedding = tf.Variable(photo_feat, trainable = False, name="i_embedding", dtype = tf.float32)
    i_vec = tf.nn.embedding_lookup(i_embedding, placeholders["Is"])
    i_vec = tf.reduce_mean(i_vec, 1)
    
    inputs = tf.concat(1, [q_vec, i_vec])
    
    # add a fully connected layer
    fc1_w = tf.get_variable("fc1_w", [inputs.get_shape()[1], num_classes], initializer=tf.random_normal_initializer())
    fc1_b = tf.get_variable("fc1_b", [num_classes], initializer=tf.random_normal_initializer())
    fc1_o = tf.nn.sigmoid(tf.matmul(inputs, fc1_w) + fc1_b)
  return fc1_o

def train_lr_embedding_q_i(loss, global_step):
  return train(loss, global_step, 0.2, optimizer = "GradientDescentOptimizer", decay_steps = 500, max_grad_norm = 5)

def build_bow(placeholders, num_classes):
  with tf.name_scope('bow'):
    inputs = tf.concat(1, [placeholders["Is"], placeholders["BoWs"]])
    fc1_w = tf.get_variable("fc1_w", [inputs.get_shape()[1], num_classes], initializer=tf.random_normal_initializer())
    fc1_b = tf.get_variable("fc1_b", [num_classes], initializer=tf.random_normal_initializer())
    fc1_o = tf.nn.sigmoid(tf.matmul(inputs, fc1_w) + fc1_b)
  return fc1_o

def train_bow(loss, global_step):
  return train(loss, global_step, 0.5, optimizer = "GradientDescentOptimizer", decay_steps = 500, max_grad_norm = 5)

def build_lstm_q(placeholders, photo_feat_file, vocabulary_size, num_classes, embedding_size = 300):
  with tf.name_scope('lstm_Q'):
    batch_size = int(placeholders["Qs"]._shape[0])
    num_steps = int(placeholders["Qs"]._shape[1])
    num_layers = 1
    
    q_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=0.0, state_is_tuple=True)] * num_layers, state_is_tuple=True)
    q_initial_state = q_cell.zero_state(batch_size, tf.float32)
    q_embedding = tf.get_variable("embedding", [vocabulary_size, embedding_size])
    q_inputs = tf.nn.embedding_lookup(q_embedding, placeholders["Qs"])
    q_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, num_steps, q_inputs)]
    with tf.variable_scope('q_lstm'):
      _, q_states = tf.nn.rnn(cell=q_cell, inputs=q_inputs, sequence_length=placeholders["Qs_l"], initial_state=q_initial_state)

    inputs = q_states[-1][1]  # get the last hidden states (dynamic length)

    fc_hidden_nodes = 512
    fc1_w = tf.get_variable("fc1_w", [inputs.get_shape()[1], fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_b = tf.get_variable("fc1_b", [fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_o = tf.nn.sigmoid(tf.matmul(inputs, fc1_w) + fc1_b)
    
    fc2_w = tf.get_variable("fc2_w", [fc_hidden_nodes, num_classes], initializer=tf.random_normal_initializer())
    fc2_b = tf.get_variable("fc2_b", [num_classes], initializer=tf.random_normal_initializer())
    fc2_o = tf.nn.relu(tf.matmul(fc1_o, fc2_w) + fc2_b)
    return fc2_o
  
def train_lstm_q(loss, global_step):
  return train(loss, global_step, 0.3, optimizer = "GradientDescentOptimizer", decay_steps = 500, max_grad_norm = 5)


def build_lstm_q_i(placeholders, photo_feat_file, vocabulary_size, num_classes, embedding_size = 300):
  with tf.name_scope('lstm_Q_I'):
    batch_size = int(placeholders["Qs"]._shape[0])
    num_steps = int(placeholders["Qs"]._shape[1])
    num_layers = 1
    
  
    q_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=0.0, state_is_tuple=True)] * num_layers, state_is_tuple=True)
    q_initial_state = q_cell.zero_state(batch_size, tf.float32)
    q_embedding = tf.get_variable("embedding", [vocabulary_size, embedding_size])
    q_inputs = tf.nn.embedding_lookup(q_embedding, placeholders["Qs"])
    q_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, num_steps, q_inputs)]
    with tf.variable_scope('q_lstm'):
      _, q_states = tf.nn.rnn(cell=q_cell, inputs=q_inputs, sequence_length=placeholders["Qs_l"], initial_state=q_initial_state)

    _, photo_feat = pickle.load(open(photo_feat_file, "rb"))
    i_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=0.0, state_is_tuple=True)] * num_layers, state_is_tuple=True)
    i_initial_state = i_cell.zero_state(batch_size, tf.float32)
    i_embedding = tf.Variable(photo_feat, trainable = False, name="i_embedding", dtype = tf.float32)
    i_inputs = tf.nn.embedding_lookup(i_embedding, placeholders["Is"])
    fci_w = tf.get_variable("fci_w", [int(i_inputs.get_shape()[2]), embedding_size], initializer=tf.random_normal_initializer())
    fci_b = tf.get_variable("fci_b", [ embedding_size], initializer=tf.random_normal_initializer())
    #fci_o = tf.matmul(fci_w, i_inputs) + fci_b
    i_inputs = [tf.nn.sigmoid(tf.matmul(tf.squeeze(t, [1]), fci_w) + fci_b)  for t in tf.split(1, num_steps, i_inputs)]
    with tf.variable_scope('i_lstm'):
      _, i_states = tf.nn.rnn(cell=i_cell, inputs=i_inputs, sequence_length=placeholders["Is_l"], initial_state=i_initial_state)

    inputs = tf.concat(1, [q_states[-1][1], i_states[-1][1]]) # get the last hidden states (dynamic length)
    
    fc_hidden_nodes = 512
    fc1_w = tf.get_variable("fc1_w", [inputs.get_shape()[1], fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_b = tf.get_variable("fc1_b", [fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_o = tf.nn.sigmoid(tf.matmul(inputs, fc1_w) + fc1_b)
    
    fc2_w = tf.get_variable("fc2_w", [fc_hidden_nodes, num_classes], initializer=tf.random_normal_initializer())
    fc2_b = tf.get_variable("fc2_b", [num_classes], initializer=tf.random_normal_initializer())
    fc2_o = tf.nn.relu(tf.matmul(fc1_o, fc2_w) + fc2_b)
    return fc2_o
  
def train_lstm_q_i(loss, global_step):
  return train(loss, global_step, 0.2, optimizer = "GradientDescentOptimizer", decay_steps = 300, max_grad_norm = 3)

  

def build_ours(q_placeholder, q_len_placeholder, i_placeholder, g_placeholder, t_placeholder, pt_placeholder, at_placeholder, vocabulary_size, num_classes, embedding_size = 100):
  """logistic regression with embedding input"""
  with tf.name_scope('ours'):
    batch_size = int(q_placeholder._shape[0])
    num_steps = int(q_placeholder._shape[1])
    
    num_layers = 2

    q_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=0.0, state_is_tuple=True)] * num_layers, state_is_tuple=True)
    pt_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=0.0, state_is_tuple=True)] * num_layers, state_is_tuple=True)
    at_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=0.0, state_is_tuple=True)] * num_layers, state_is_tuple=True)
    
    q_initial_state = q_cell.zero_state(batch_size, tf.float32)
    pt_initial_state = pt_cell.zero_state(batch_size, tf.float32)
    at_initial_state = at_cell.zero_state(batch_size, tf.float32)



    embedding = tf.get_variable("embedding", [vocabulary_size, embedding_size])
    q_inputs = tf.nn.embedding_lookup(embedding, q_placeholder)
    q_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, num_steps, q_inputs)]
    with tf.variable_scope('q_lstm'):
      _, q_states = tf.nn.rnn(q_cell, q_inputs, initial_state=q_initial_state)
      
    pt_inputs = tf.nn.embedding_lookup(embedding, pt_placeholder)
    pt_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, num_steps, pt_inputs)]
    with tf.variable_scope('pt_lstm'):
      _, pt_states = tf.nn.rnn(pt_cell, pt_inputs, initial_state=pt_initial_state)

    at_inputs = tf.nn.embedding_lookup(embedding, at_placeholder)
    at_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, num_steps, at_inputs)]
    with tf.variable_scope('at_lstm'):
      _, at_states = tf.nn.rnn(pt_cell, at_inputs, initial_state=at_initial_state)    
    
    #states = states[num_steps,] # get the last states

#     q_embedding = tf.nn.embedding_lookup(embedding, q_placeholder)
#     q_embedding = tf.reduce_max(q_embedding, 1)
#        
#     g_embedding = tf.nn.embedding_lookup(embedding, g_placeholder)
#     g_embedding = tf.reduce_max(g_embedding, 1)
#        
    t_embedding = tf.nn.embedding_lookup(embedding, t_placeholder)
    t_embedding = tf.reduce_mean(t_embedding, 1)
      
    
    #inputs = tf.concat(1, [t_embedding, q_outputs[num_steps-1], pt_outputs[num_steps-1], at_outputs[num_steps-1]])
    inputs = q_states[-1][1]  # get the last hidden states (dynamic length)


    
    fc_hidden_nodes = 1024
    fc1_w = tf.get_variable("fc1_w", [inputs.get_shape()[1], fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_b = tf.get_variable("fc1_b", [fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_o = tf.nn.sigmoid(tf.matmul(inputs, fc1_w) + fc1_b)
    
    fc2_w = tf.get_variable("fc2_w", [fc_hidden_nodes, num_classes], initializer=tf.random_normal_initializer())
    fc2_b = tf.get_variable("fc2_b", [num_classes], initializer=tf.random_normal_initializer())
    fc2_o = tf.nn.relu(tf.matmul(fc1_o, fc2_w) + fc2_b)

  return fc2_o



def calculate_xentropy_loss(outputs, labels):
  with tf.name_scope('loss'):
    label_vector = tf.to_float(labels)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(outputs, label_vector)
    loss = tf.reduce_mean(cross_entropy)
    return loss

def calculate_softmax_loss(outputs, labels):
  with tf.name_scope('loss'):
    label_vector = tf.to_float(labels)
    softmax = tf.nn.softmax_cross_entropy_with_logits(outputs, label_vector)
    loss = tf.reduce_mean(softmax)
  return loss

def train(loss, global_step, starter_learning_rate, optimizer = "GradientDescentOptimizer", decay_steps = 1000, max_grad_norm = 5):
  with tf.name_scope('train'):
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, 0.96, staircase=True)
    tf.add_to_collection("lr", learning_rate)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
    optimizer_op = getattr(tf.train, optimizer)(learning_rate) # tune the optimizer GradientDescentOptimizer AdadeltaOptimizer
    with tf.control_dependencies([loss]):
      train_op = optimizer_op.apply_gradients(zip(grads, tvars), global_step=global_step)
  return train_op
