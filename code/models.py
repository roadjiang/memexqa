'''
Created on Mar 10, 2017

@author: lujiang
'''

import tensorflow as tf
import pickle



def build_lr(placeholders, photo_feat_file, vocabulary_size, num_classes, embedding_size = 100):
  """logistic regression with embedding input"""
  with tf.name_scope('lr'):
    
    word2vec = pickle.load(open("/Users/lujiang/data/memex_dataset/exp/word2vec_vocabulary_embedding.p", "rb"))
    word_embedding = tf.Variable(word2vec, trainable = False, dtype = tf.float32)
    q_vec = tf.nn.embedding_lookup(word_embedding, placeholders["Qs"])
    q_vec = tf.reduce_mean(q_vec, 1)
    
    meta_vec = tf.nn.embedding_lookup(word_embedding, placeholders["PTs"])
    meta_vec = tf.reduce_mean(meta_vec, 1)

    _, photo_feat = pickle.load(open(photo_feat_file, "rb"))
    i_embedding = tf.Variable(photo_feat, trainable = False, name="i_embedding", dtype = tf.float32)
    i_vec = tf.nn.embedding_lookup(i_embedding, placeholders["Is"])
    i_vec = tf.reduce_mean(i_vec, 1)
    
    inputs = tf.concat(1, [q_vec, meta_vec, i_vec])
    
    # add a fully connected layer
    fc1_w = tf.get_variable("fc1_w", [inputs.get_shape()[1], num_classes], initializer=tf.random_normal_initializer())
    fc1_b = tf.get_variable("fc1_b", [num_classes], initializer=tf.random_normal_initializer())
    fc1_o = tf.nn.sigmoid(tf.matmul(inputs, fc1_w) + fc1_b)
  return fc1_o

def train_lr(loss, global_step):
  return train(loss, global_step, 0.3, optimizer = "GradientDescentOptimizer", decay_steps = 300, max_grad_norm = 5)


def build_embedding(placeholders, photo_feat_file, vocabulary_size, num_classes, embedding_size = 100):
  """logistic regression with embedding input"""
  with tf.name_scope('lr'):
    
    word_embedding = tf.get_variable("word_embedding", [vocabulary_size, embedding_size])
    q_vec = tf.nn.embedding_lookup(word_embedding, placeholders["Qs"])
    q_vec = tf.reduce_mean(q_vec, 1)
    
    meta_vec = tf.nn.embedding_lookup(word_embedding, placeholders["PTs"])
    meta_vec = tf.reduce_mean(meta_vec, 1)

    _, photo_feat = pickle.load(open(photo_feat_file, "rb"))
    i_embedding = tf.Variable(photo_feat, trainable = False, name="i_embedding", dtype = tf.float32)
    i_vec = tf.nn.embedding_lookup(i_embedding, placeholders["Is"])
    i_vec = tf.reduce_mean(i_vec, 1)
    
    inputs = tf.concat(1, [q_vec, meta_vec, i_vec])
    
    # add a fully connected layer
    fc1_w = tf.get_variable("fc1_w", [inputs.get_shape()[1], num_classes], initializer=tf.random_normal_initializer())
    fc1_b = tf.get_variable("fc1_b", [num_classes], initializer=tf.random_normal_initializer())
    fc1_o = tf.nn.sigmoid(tf.matmul(inputs, fc1_w) + fc1_b)
  return fc1_o

def train_embedding(loss, global_step):
  return train(loss, global_step, 0.3, optimizer = "GradientDescentOptimizer", decay_steps = 300, max_grad_norm = 5)



def build_bow(placeholders, num_classes):
  with tf.name_scope('bow'):
    inputs = tf.concat(1, [placeholders["Is"], placeholders["BoWs"]])
    fc1_w = tf.get_variable("fc1_w", [inputs.get_shape()[1], num_classes], initializer=tf.random_normal_initializer())
    fc1_b = tf.get_variable("fc1_b", [num_classes], initializer=tf.random_normal_initializer())
    fc1_o = tf.nn.sigmoid(tf.matmul(inputs, fc1_w) + fc1_b)
  return fc1_o

def train_bow(loss, global_step):
  return train(loss, global_step, 0.5, optimizer = "GradientDescentOptimizer", decay_steps = 500, max_grad_norm = 5)

def build_lstm(placeholders, photo_feat_file, vocabulary_size, num_classes, embedding_size = 100):
  with tf.name_scope('lstm'):
    batch_size = int(placeholders["Qs"]._shape[0])
    num_steps = int(placeholders["Qs"]._shape[1])
    num_layers = 1
    

    word_embedding = tf.get_variable("word_embedding", [vocabulary_size, embedding_size])
    q_inputs = tf.nn.embedding_lookup(word_embedding, placeholders["Qs"])
    
    meta_inputs = tf.nn.embedding_lookup(word_embedding, placeholders["PTs"])
    meta_inputs = tf.reduce_mean(meta_inputs, 1)
    
    _, photo_feat = pickle.load(open(photo_feat_file, "rb"))
    i_embedding = tf.Variable(photo_feat, trainable = False, name="i_embedding", dtype = tf.float32)
    i_inputs = tf.nn.embedding_lookup(i_embedding, placeholders["Is"])
    i_inputs = tf.reduce_mean(i_inputs, 1)
    
    
    
    q_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=0.0, state_is_tuple=True)] * num_layers, state_is_tuple=True)
    q_initial_state = q_cell.zero_state(batch_size, tf.float32)
    
     
    q_inputs = [tf.concat(1, [tf.squeeze(t, [1]), meta_inputs, i_inputs]) for t in tf.split(1, num_steps, q_inputs)]
    with tf.variable_scope('lstm'):
      _, q_states = tf.nn.rnn(cell=q_cell, inputs=q_inputs, sequence_length=placeholders["Qs_l"], initial_state=q_initial_state)

    #inputs = tf.concat(1, [q_states[-1][1], i_states[-1][1]]) # get the last hidden states (dynamic length)
    inputs = q_states[-1][1]
    
    fc_hidden_nodes = 32
    fc1_w = tf.get_variable("fc1_w", [inputs.get_shape()[1], fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_b = tf.get_variable("fc1_b", [fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_o = tf.nn.sigmoid(tf.matmul(inputs, fc1_w) + fc1_b)
    
    fc2_w = tf.get_variable("fc2_w", [fc_hidden_nodes, num_classes], initializer=tf.random_normal_initializer())
    fc2_b = tf.get_variable("fc2_b", [num_classes], initializer=tf.random_normal_initializer())
    fc2_o = tf.nn.relu(tf.matmul(fc1_o, fc2_w) + fc2_b)
    return fc2_o
  
def train_lstm(loss, global_step):
  return train(loss, global_step, 0.3, optimizer = "AdagradOptimizer", decay_steps = 300, max_grad_norm = 3)


def build_lstm_nolookup(placeholders, photo_feat_file, vocabulary_size, num_classes, embedding_size = 100):
  with tf.name_scope('lstm'):
    batch_size = int(placeholders["Qs"]._shape[0])
    num_steps = int(placeholders["Qs"]._shape[1])
    num_layers = 1
    

    word_embedding = tf.get_variable("word_embedding", [vocabulary_size, embedding_size])
    q_inputs = tf.nn.embedding_lookup(word_embedding, placeholders["Qs"])
    
    meta_inputs = tf.nn.embedding_lookup(word_embedding, placeholders["PTs"])
    meta_inputs = tf.reduce_mean(meta_inputs, 1)
    
    _, photo_feat = pickle.load(open(photo_feat_file, "rb"))
    i_embedding = tf.Variable(photo_feat, trainable = False, name="i_embedding", dtype = tf.float32)
    i_inputs = tf.nn.embedding_lookup(i_embedding, placeholders["Is"])
    i_inputs = tf.reduce_mean(i_inputs, 1)
    
    
    
    q_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=0.0, state_is_tuple=True)] * num_layers, state_is_tuple=True)
    q_initial_state = q_cell.zero_state(batch_size, tf.float32)
    
     
    q_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, num_steps, q_inputs)]
    with tf.variable_scope('lstm'):
      _, q_states = tf.nn.rnn(cell=q_cell, inputs=q_inputs, sequence_length=placeholders["Qs_l"], initial_state=q_initial_state)

    inputs = tf.concat(1, [q_states[-1][1], i_inputs, meta_inputs]) # get the last hidden states (dynamic length)
    
    fc_hidden_nodes = 32
    fc1_w = tf.get_variable("fc1_w", [inputs.get_shape()[1], fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_b = tf.get_variable("fc1_b", [fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_o = tf.nn.sigmoid(tf.matmul(inputs, fc1_w) + fc1_b)
    
    fc2_w = tf.get_variable("fc2_w", [fc_hidden_nodes, num_classes], initializer=tf.random_normal_initializer())
    fc2_b = tf.get_variable("fc2_b", [num_classes], initializer=tf.random_normal_initializer())
    fc2_o = tf.nn.relu(tf.matmul(fc1_o, fc2_w) + fc2_b)
    return fc2_o
  
def train_lstm_nolookup(loss, global_step):
  return train(loss, global_step, 0.3, optimizer = "AdagradOptimizer", decay_steps = 500, max_grad_norm = 3)


def build_lstm_att(placeholders, photo_feat_file, vocabulary_size, num_classes, embedding_size = 100):
  with tf.name_scope('lstm'):
    batch_size = int(placeholders["Qs"]._shape[0])
    num_steps = int(placeholders["Qs"]._shape[1])
    num_layers = 1
    

    word_embedding = tf.get_variable("word_embedding", [vocabulary_size, embedding_size])
    q_inputs = tf.nn.embedding_lookup(word_embedding, placeholders["Qs"])
    
    meta_inputs = tf.nn.embedding_lookup(word_embedding, placeholders["PTs"])
    meta_inputs = tf.reduce_mean(meta_inputs, 1)
    
    _, photo_feat = pickle.load(open(photo_feat_file, "rb"))
    i_embedding = tf.Variable(photo_feat, trainable = False, name="i_embedding", dtype = tf.float32)
    i_inputs = tf.nn.embedding_lookup(i_embedding, placeholders["Is"])
    i_inputs = tf.reduce_mean(i_inputs, 1)
    
    
    
    q_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=0.0, state_is_tuple=True)] * num_layers, state_is_tuple=True)
    #q_cell = tf.contrib.rnn.AttentionCellWrapper(q_cell, attn_length=8, attn_size=2, attn_vec_size=10, state_is_tuple=True)    
    
    
    q_initial_state = q_cell.zero_state(batch_size, tf.float32)
    
     
    q_inputs = [tf.concat(1, [tf.squeeze(t, [1]), meta_inputs, i_inputs]) for t in tf.split(1, num_steps, q_inputs)]
    with tf.variable_scope('lstm'):
      _, q_states = tf.nn.rnn(cell=q_cell, inputs=q_inputs, sequence_length=placeholders["Qs_l"], initial_state=q_initial_state)

    #inputs = tf.concat(1, [q_states[-1][1], i_states[-1][1]]) # get the last hidden states (dynamic length)
    inputs = q_states[-1][1]
    
    fc_hidden_nodes = 32
    fc1_w = tf.get_variable("fc1_w", [inputs.get_shape()[1], fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_b = tf.get_variable("fc1_b", [fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_o = tf.nn.sigmoid(tf.matmul(inputs, fc1_w) + fc1_b)
    
    fc2_w = tf.get_variable("fc2_w", [fc_hidden_nodes, num_classes], initializer=tf.random_normal_initializer())
    fc2_b = tf.get_variable("fc2_b", [num_classes], initializer=tf.random_normal_initializer())
    fc2_o = tf.nn.relu(tf.matmul(fc1_o, fc2_w) + fc2_b)
    return fc2_o
  
def train_lstm_att(loss, global_step):
  return train(loss, global_step, 0.3, optimizer = "GradientDescentOptimizer", decay_steps = 300, max_grad_norm = 3)

def build_lstm_mc(placeholders, photo_feat_file, vocabulary_size, num_classes, embedding_size = 100):
  with tf.name_scope('lstm_mc'):
    batch_size = int(placeholders["Qs"]._shape[0])
    num_steps = int(placeholders["Qs"]._shape[1])
    num_layers = 1
    
    q_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=0.0, state_is_tuple=True)] * num_layers, state_is_tuple=True)
    q_initial_state = q_cell.zero_state(batch_size, tf.float32)
    word_embedding = tf.get_variable("word_embedding", [vocabulary_size, embedding_size])
    q_inputs = tf.nn.embedding_lookup(word_embedding, placeholders["Qs"])
    
    
    q_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, num_steps, q_inputs)]
    with tf.variable_scope('q_lstm'):
      _, q_states = tf.nn.rnn(cell=q_cell, inputs=q_inputs, sequence_length=placeholders["Qs_l"], initial_state=q_initial_state)

    _, photo_feat = pickle.load(open(photo_feat_file, "rb"))
    i_embedding = tf.Variable(photo_feat, trainable = False, name="i_embedding", dtype = tf.float32)
    
    
    i_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=0.0, state_is_tuple=True)] * num_layers, state_is_tuple=True)
    i_initial_state = i_cell.zero_state(batch_size, tf.float32)
    i_inputs = tf.nn.embedding_lookup(i_embedding, placeholders["Is"])
    
    fci_w = tf.get_variable("fci_w", [int(i_inputs.get_shape()[2]), embedding_size], initializer=tf.random_normal_initializer())
    fci_b = tf.get_variable("fci_b", [ embedding_size], initializer=tf.random_normal_initializer())
    #fci_o = tf.matmul(fci_w, i_inputs) + fci_b
    i_inputs = [tf.nn.sigmoid(tf.matmul(tf.squeeze(t, [1]), fci_w) + fci_b)  for t in tf.split(1, num_steps, i_inputs)]
    with tf.variable_scope('i_lstm'):
      _, i_states = tf.nn.rnn(cell=i_cell, inputs=i_inputs, sequence_length=placeholders["Is_l"], initial_state=i_initial_state)

    m_inputs = tf.nn.embedding_lookup(word_embedding, placeholders["PTs"])
    m_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, num_steps, m_inputs)]
    with tf.variable_scope('m_lstm'):
      _, m_states = tf.nn.rnn(cell=q_cell, inputs=m_inputs, sequence_length=placeholders["PTs_l"], initial_state=q_initial_state)

    #inputs = tf.concat(1, [q_states[-1][1], i_states[-1][1]]) # get the last hidden states (dynamic length)
    inputs = tf.mul(tf.mul(q_states[-1][1], i_states[-1][1]), m_states[-1][1])
    
    fc_hidden_nodes = 32
    fc1_w = tf.get_variable("fc1_w", [inputs.get_shape()[1], fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_b = tf.get_variable("fc1_b", [fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_o = tf.nn.sigmoid(tf.matmul(inputs, fc1_w) + fc1_b)
    
    fc2_w = tf.get_variable("fc2_w", [fc_hidden_nodes, num_classes], initializer=tf.random_normal_initializer())
    fc2_b = tf.get_variable("fc2_b", [num_classes], initializer=tf.random_normal_initializer())
    fc2_o = tf.nn.relu(tf.matmul(fc1_o, fc2_w) + fc2_b)
    return fc2_o
  
def train_lstm_mc(loss, global_step):
  return train(loss, global_step, 0.3, optimizer = "AdagradOptimizer", decay_steps = 1000, max_grad_norm = 3)



def build_memex_lookup1(placeholders, photo_feat_file, vocabulary_size, num_classes, embedding_size = 100):
  with tf.name_scope('memex_lookup'):
    batch_size = int(placeholders["Qs"]._shape[0])
    q_num_steps = int(placeholders["Qs"]._shape[1])
    
    num_layers = 1
    
    text_embedding = tf.get_variable("text_embedding", [vocabulary_size, embedding_size])

    q_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=0.0, state_is_tuple=True)] * num_layers, state_is_tuple=True)
    q_initial_state = q_cell.zero_state(batch_size, tf.float32)
    q_inputs = tf.nn.embedding_lookup(text_embedding, placeholders["Qs"])
    q_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, q_num_steps, q_inputs)]

    with tf.variable_scope('q_lstm'):
      _, q_states = tf.nn.rnn(cell=q_cell, inputs=q_inputs, sequence_length=placeholders["Qs_l"], initial_state=q_initial_state)
      #q_outputs, _ = tf.nn.rnn(cell=q_cell, inputs=q_inputs, initial_state=q_initial_state)
      
    
      
    
#     _, photo_feat = pickle.load(open(photo_feat_file, "rb"))
#     image_embedding = tf.Variable(photo_feat, trainable = False, name="image_embedding", dtype = tf.float32)
#     i_inputs = tf.nn.embedding_lookup(image_embedding, placeholders["Is"])
#     
#     fci_w = tf.get_variable("fci_w", [int(i_inputs.get_shape()[1]), embedding_size], initializer=tf.random_normal_initializer())
#     fci_b = tf.get_variable("fci_b", [embedding_size], initializer=tf.random_normal_initializer())
#     i_outputs =  tf.matmul(i_inputs, fci_w) + fci_b
    
    mm_embedding_size = 10
    class_embedding = tf.get_variable("class_embedding", [num_classes, 10])
    
    pts_matched = tf.nn.embedding_lookup(class_embedding, placeholders["PTs"])
    #ts_matched = tf.nn.embedding_lookup(class_embedding, placeholders["Ts"])
    
    pts_matched = tf.reshape(pts_matched, [batch_size, mm_embedding_size * int(placeholders["PTs"].get_shape()[1])])
    
    const_ones = tf.ones([batch_size, 1], tf.float32)
    
    t_embed = tf.get_variable("t_embed", [1, mm_embedding_size], initializer=tf.random_normal_initializer())
    #g_embed = tf.get_variable("g_embed", [1, mm_embedding_size], initializer=tf.random_normal_initializer())
    pt_embed = tf.get_variable("pt_embed", [1, mm_embedding_size], initializer=tf.random_normal_initializer())
    
    t_feat =  tf.matmul(const_ones, t_embed)
    #g_feat =  tf.matmul(const_ones, g_embed)
    pt_feat = tf.matmul(const_ones, pt_embed)
    
    q_hidden = q_states[-1][1]
    
    #fc0_w = tf.get_variable("fc0_w", [q_hidden.get_shape()[1], 50], initializer=tf.random_normal_initializer())
    #fc0_b = tf.get_variable("fc0_b", [50], initializer=tf.random_normal_initializer())
    #fc0_o = tf.nn.relu(tf.matmul(q_hidden, fc0_w) + fc0_b)
    
    #inputs = tf.concat(1, [q_outputs[-1], i_outputs[-1]])
    inputs = tf.concat(1, [q_hidden, pts_matched])
    #inputs = tf.concat(1, [q_states[-1][1], placeholders["Ts"], placeholders["PTs"]])

    #inputs = q_states[-1][1]
    
    fc_hidden_nodes = 32
    fc1_w = tf.get_variable("fc1_w", [inputs.get_shape()[1], fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_b = tf.get_variable("fc1_b", [fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_o = tf.nn.sigmoid(tf.matmul(inputs, fc1_w) + fc1_b)
    
    fc2_w = tf.get_variable("fc2_w", [fc_hidden_nodes, num_classes], initializer=tf.random_normal_initializer())
    fc2_b = tf.get_variable("fc2_b", [num_classes], initializer=tf.random_normal_initializer())
    fc2_o = tf.nn.relu(tf.matmul(fc1_o, fc2_w) + fc2_b)
  return fc2_o



def train_memex_lookup1(loss, global_step):
  return train(loss, global_step, 0.3, optimizer = "AdagradOptimizer", decay_steps =1000, max_grad_norm = 5)


def build_memexnet_lookup2(placeholders, photo_feat_file, vocabulary_size, num_classes, embedding_size = 100):
  with tf.name_scope('memex_lookup'):
    batch_size = int(placeholders["Qs"]._shape[0])
    q_num_steps = int(placeholders["Qs"]._shape[1])
    
    num_layers = 1
    
    text_embedding = tf.get_variable("text_embedding", [vocabulary_size, embedding_size])

    q_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=0.0, state_is_tuple=True)] * num_layers, state_is_tuple=True)
    q_initial_state = q_cell.zero_state(batch_size, tf.float32)
    q_inputs = tf.nn.embedding_lookup(text_embedding, placeholders["Qs"])
    q_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, q_num_steps, q_inputs)]

    with tf.variable_scope('q_lstm'):
      _, q_states = tf.nn.rnn(cell=q_cell, inputs=q_inputs, sequence_length=placeholders["Qs_l"], initial_state=q_initial_state)
      #q_outputs, _ = tf.nn.rnn(cell=q_cell, inputs=q_inputs, initial_state=q_initial_state)
      
    
    mm_embedding_size = 10
    class_embedding = tf.get_variable("class_embedding", [num_classes, mm_embedding_size])
    
    pts_matched = tf.nn.embedding_lookup(class_embedding, placeholders["PTs"])
    ts_matched = tf.nn.embedding_lookup(class_embedding, placeholders["Ts"])
    
    pts_matched = tf.reshape(pts_matched, [batch_size, mm_embedding_size * int(placeholders["PTs"].get_shape()[1])])
    ts_matched = tf.reshape(ts_matched, [batch_size, mm_embedding_size * int(placeholders["Ts"].get_shape()[1])])

    
    const_ones = tf.ones([batch_size, 1], tf.float32)
    
    t_embed = tf.get_variable("t_embed", [1, 1], initializer=tf.random_normal_initializer())
    #g_embed = tf.get_variable("g_embed", [1, mm_embedding_size], initializer=tf.random_normal_initializer())
    pt_embed = tf.get_variable("pt_embed", [1, mm_embedding_size], initializer=tf.random_normal_initializer())
    
    mm_t =  tf.matmul(const_ones, t_embed)
    #g_feat =  tf.matmul(const_ones, g_embed)
    mm_pt = tf.matmul(const_ones, pt_embed)
    
    q_hidden = q_states[-1][1]
    
    #fc0_w = tf.get_variable("fc0_w", [q_hidden.get_shape()[1], 50], initializer=tf.random_normal_initializer())
    #fc0_b = tf.get_variable("fc0_b", [50], initializer=tf.random_normal_initializer())
    #fc0_o = tf.nn.relu(tf.matmul(q_hidden, fc0_w) + fc0_b)
    
    #inputs = tf.concat(1, [q_outputs[-1], i_outputs[-1]])
    inputs = tf.concat(1, [q_hidden, mm_pt, pts_matched, mm_t, ts_matched])
    #inputs = tf.concat(1, [q_states[-1][1], placeholders["Ts"], placeholders["PTs"]])

    #inputs = q_states[-1][1]
    
    fc_hidden_nodes = 32
    fc1_w = tf.get_variable("fc1_w", [inputs.get_shape()[1], fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_b = tf.get_variable("fc1_b", [fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_o = tf.nn.sigmoid(tf.matmul(inputs, fc1_w) + fc1_b)
    
    fc2_w = tf.get_variable("fc2_w", [fc_hidden_nodes, num_classes], initializer=tf.random_normal_initializer())
    fc2_b = tf.get_variable("fc2_b", [num_classes], initializer=tf.random_normal_initializer())
    fc2_o = tf.nn.relu(tf.matmul(fc1_o, fc2_w) + fc2_b)
  return fc2_o



def train_memexnet_img(loss, global_step):
  return train(loss, global_step, 0.3, optimizer = "AdagradOptimizer", decay_steps =1000, max_grad_norm = 5)


def build_memexnet_img(placeholders, photo_feat_file, vocabulary_size, num_classes, embedding_size = 100):
  with tf.name_scope('memex_lookup'):
    batch_size = int(placeholders["Qs"]._shape[0])
    q_num_steps = int(placeholders["Qs"]._shape[1])
    
    num_layers = 1
    
    text_embedding = tf.get_variable("text_embedding", [vocabulary_size, embedding_size])

    q_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=0.0, state_is_tuple=True)] * num_layers, state_is_tuple=True)
    q_initial_state = q_cell.zero_state(batch_size, tf.float32)
    q_inputs = tf.nn.embedding_lookup(text_embedding, placeholders["Qs"])
    q_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, q_num_steps, q_inputs)]

    with tf.variable_scope('q_lstm'):
      _, q_states = tf.nn.rnn(cell=q_cell, inputs=q_inputs, sequence_length=placeholders["Qs_l"], initial_state=q_initial_state)
      #q_outputs, _ = tf.nn.rnn(cell=q_cell, inputs=q_inputs, initial_state=q_initial_state)
      
    
    mm_embedding_size = 10
    class_embedding = tf.get_variable("class_embedding", [num_classes, mm_embedding_size])
    
    pts_matched = tf.nn.embedding_lookup(class_embedding, placeholders["PTs"])
    ts_matched = tf.nn.embedding_lookup(class_embedding, placeholders["Ts"])
    
    pts_matched = tf.reshape(pts_matched, [batch_size, mm_embedding_size * int(placeholders["PTs"].get_shape()[1])])
    ts_matched = tf.reshape(ts_matched, [batch_size, mm_embedding_size * int(placeholders["Ts"].get_shape()[1])])

    _, photo_feat = pickle.load(open(photo_feat_file, "rb"))
    image_embedding = tf.Variable(photo_feat, trainable = False, name="image_embedding", dtype = tf.float32)
    i_inputs = tf.nn.embedding_lookup(image_embedding, placeholders["Is"])
    
    const_ones = tf.ones([batch_size, 1], tf.float32)
    
    t_embed = tf.get_variable("t_embed", [1, 1], initializer=tf.random_normal_initializer())
    #g_embed = tf.get_variable("g_embed", [1, mm_embedding_size], initializer=tf.random_normal_initializer())
    pt_embed = tf.get_variable("pt_embed", [1, mm_embedding_size], initializer=tf.random_normal_initializer())
    
    mm_t =  tf.matmul(const_ones, t_embed)
    #g_feat =  tf.matmul(const_ones, g_embed)
    mm_pt = tf.matmul(const_ones, pt_embed)
    
    q_hidden = q_states[-1][1]
    
    #fc0_w = tf.get_variable("fc0_w", [q_hidden.get_shape()[1], 50], initializer=tf.random_normal_initializer())
    #fc0_b = tf.get_variable("fc0_b", [50], initializer=tf.random_normal_initializer())
    #fc0_o = tf.nn.relu(tf.matmul(q_hidden, fc0_w) + fc0_b)
    
    #inputs = tf.concat(1, [q_outputs[-1], i_outputs[-1]])
    inputs = tf.concat(1, [q_hidden,  mm_pt, pts_matched, mm_t, ts_matched])
  
    #inputs = tf.concat(1, [q_states[-1][1], placeholders["Ts"], placeholders["PTs"]])

    #inputs = q_states[-1][1]
    
    fc_hidden_nodes = 32
    fc1_w = tf.get_variable("fc1_w", [inputs.get_shape()[1], fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_b = tf.get_variable("fc1_b", [fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_o = tf.nn.sigmoid(tf.matmul(inputs, fc1_w) + fc1_b)
    
    fc2_w = tf.get_variable("fc2_w", [fc_hidden_nodes, num_classes], initializer=tf.random_normal_initializer())
    fc2_b = tf.get_variable("fc2_b", [num_classes], initializer=tf.random_normal_initializer())
    fc2_o = tf.nn.relu(tf.matmul(fc1_o, fc2_w) + fc2_b)
  return fc2_o



def train_memexnet_lookup2(loss, global_step):
  return train(loss, global_step, 0.3, optimizer = "AdagradOptimizer", decay_steps =1000, max_grad_norm = 5)

def build_memexnet(placeholders, photo_feat_file, vocabulary_size, num_classes, embedding_size = 100):
  with tf.name_scope('memex_lookup'):
    batch_size = int(placeholders["Qs"]._shape[0])
    q_num_steps = int(placeholders["Qs"]._shape[1])
    
    num_layers = 1
    
    text_embedding = tf.get_variable("text_embedding", [vocabulary_size, embedding_size])

    q_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=0.0, state_is_tuple=True)] * num_layers, state_is_tuple=True)
    q_initial_state = q_cell.zero_state(batch_size, tf.float32)
    q_inputs = tf.nn.embedding_lookup(text_embedding, placeholders["Qs"])
    q_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, q_num_steps, q_inputs)]

    with tf.variable_scope('q_lstm'):
      _, q_states = tf.nn.rnn(cell=q_cell, inputs=q_inputs, sequence_length=placeholders["Qs_l"], initial_state=q_initial_state)
      #q_outputs, _ = tf.nn.rnn(cell=q_cell, inputs=q_inputs, initial_state=q_initial_state)
      
    
    
    _, photo_feat = pickle.load(open(photo_feat_file, "rb"))
    image_embedding = tf.Variable(photo_feat, trainable = False, name="image_embedding", dtype = tf.float32)
    i_inputs = tf.nn.embedding_lookup(image_embedding, placeholders["Is"])
     
    fci_w = tf.get_variable("fci_w", [int(i_inputs.get_shape()[1]), 100], initializer=tf.random_normal_initializer())
    fci_b = tf.get_variable("fci_b", [100], initializer=tf.random_normal_initializer())
    i_outputs =  tf.matmul(i_inputs, fci_w) + fci_b
    
    
    mm_embedding_size = 10
    class_embedding = tf.get_variable("class_embedding", [num_classes, mm_embedding_size])
    
    pts_matched = tf.nn.embedding_lookup(class_embedding, placeholders["PTs"])
    ts_matched = tf.nn.embedding_lookup(class_embedding, placeholders["Ts"])
    
    pts_matched = tf.reshape(pts_matched, [batch_size, mm_embedding_size * int(placeholders["PTs"].get_shape()[1])])
    ts_matched = tf.reshape(ts_matched, [batch_size, mm_embedding_size * int(placeholders["Ts"].get_shape()[1])])

    
    const_ones = tf.ones([batch_size, 1], tf.float32)
    
    t_embed = tf.get_variable("t_embed", [1, 1], initializer=tf.random_normal_initializer())
    #g_embed = tf.get_variable("g_embed", [1, mm_embedding_size], initializer=tf.random_normal_initializer())
    pt_embed = tf.get_variable("pt_embed", [1, mm_embedding_size], initializer=tf.random_normal_initializer())
    
    mm_t =  tf.matmul(const_ones, t_embed)
    #g_feat =  tf.matmul(const_ones, g_embed)
    mm_pt = tf.matmul(const_ones, pt_embed)
    
    q_hidden = q_states[-1][1]
    #q_hidden = tf.nn.dropout(q_hidden, 0.95)
    #fc0_w = tf.get_variable("fc0_w", [q_hidden.get_shape()[1], 50], initializer=tf.random_normal_initializer())
    #fc0_b = tf.get_variable("fc0_b", [50], initializer=tf.random_normal_initializer())
    #fc0_o = tf.nn.relu(tf.matmul(q_hidden, fc0_w) + fc0_b)
    
    #inputs = tf.concat(1, [q_outputs[-1], i_outputs[-1]])
    inputs = tf.concat(1, [q_hidden, mm_pt, pts_matched, mm_t, ts_matched])
    #inputs = tf.concat(1, [q_states[-1][1], placeholders["Ts"], placeholders["PTs"]])

    #inputs = q_states[-1][1]
    
    fc_hidden_nodes = 32
    fc1_w = tf.get_variable("fc1_w", [inputs.get_shape()[1], fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_b = tf.get_variable("fc1_b", [fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_o = tf.nn.sigmoid(tf.matmul(inputs, tf.nn.softmax(fc1_w)) + fc1_b)
    
    fc2_w = tf.get_variable("fc2_w", [fc_hidden_nodes, num_classes], initializer=tf.random_normal_initializer())
    fc2_b = tf.get_variable("fc2_b", [num_classes], initializer=tf.random_normal_initializer())
    fc2_o = tf.nn.relu(tf.matmul(fc1_o, fc2_w) + fc2_b)
  return fc2_o



def train_memexnet(loss, global_step):
  return train(loss, global_step, 0.3, optimizer = "AdagradOptimizer", decay_steps =500, max_grad_norm = 5)

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

def calculate_softmax_loss_4_choices(outputs, labels, mask):
  with tf.name_scope('loss'):
    label_vector = tf.to_float(labels)
    outputs = tf.mul(outputs, mask)
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

def build_lr_embedding_q_i(placeholders, photo_feat_file, vocabulary_size, num_classes, embedding_size = 100):
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
  return train(loss, global_step, 0.2, optimizer = "GradientDescentOptimizer", decay_steps = 300, max_grad_norm = 5)


def build_lstm_q(placeholders, photo_feat_file, vocabulary_size, num_classes, embedding_size = 300):
  with tf.name_scope('lstm'):
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
  return train(loss, global_step, 0.3, optimizer = "GradientDescentOptimizer", decay_steps = 300, max_grad_norm = 5)


def build_lstm_q_i(placeholders, photo_feat_file, vocabulary_size, num_classes, embedding_size = 100):
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

    #inputs = tf.concat(1, [q_states[-1][1], i_states[-1][1]]) # get the last hidden states (dynamic length)
    inputs = tf.mul(q_states[-1][1], i_states[-1][1])
    
    fc_hidden_nodes = 32
    fc1_w = tf.get_variable("fc1_w", [inputs.get_shape()[1], fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_b = tf.get_variable("fc1_b", [fc_hidden_nodes], initializer=tf.random_normal_initializer())
    fc1_o = tf.nn.sigmoid(tf.matmul(inputs, fc1_w) + fc1_b)
    
    fc2_w = tf.get_variable("fc2_w", [fc_hidden_nodes, num_classes], initializer=tf.random_normal_initializer())
    fc2_b = tf.get_variable("fc2_b", [num_classes], initializer=tf.random_normal_initializer())
    fc2_o = tf.nn.relu(tf.matmul(fc1_o, fc2_w) + fc2_b)
    return fc2_o
  
def train_lstm_q_i(loss, global_step):
  return train(loss, global_step, 0.2, optimizer = "GradientDescentOptimizer", decay_steps = 300, max_grad_norm = 3)
