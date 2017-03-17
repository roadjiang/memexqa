'''
Created on Mar 10, 2017

@author: lujiang
'''
import time
import os

import numpy as np
import models
import tensorflow as tf
import logging

import eval as myeval
import reader_bow


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
 
flags = tf.flags
flags.DEFINE_string("train_dir", "/Users/lujiang/run/bow_tr", "Training output directory")
flags.DEFINE_string("data_path", "/Users/lujiang/data/memex_dataset/exp/bow_tr.p", "data_path")


flags.DEFINE_integer("batch_size", 32, "training batch size")
flags.DEFINE_float("start_learning_rate", 0.2, "starting learning rate")
flags.DEFINE_boolean("retrain", False, "whether to retrain or not")

FLAGS = flags.FLAGS


def train(loss, global_step, starter_learning_rate, max_grad_norm = 5):
  with tf.name_scope('train'):
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
    tf.add_to_collection("lr", learning_rate)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
    optimizer = tf.train.AdagradOptimizer(learning_rate) # tune the optimizer GradientDescentOptimizer AdadeltaOptimizer
    with tf.control_dependencies([loss]):
      train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
  return train_op


def update_iter_info(summary_writer, iter_info):
  iter_id = iter_info["global_step"]
  summary_writer.add_summary(make_summary("iteration/" + "f1", iter_info["accuracy"]), iter_id)
  summary_writer.add_summary(make_summary("iteration/" + "loss", iter_info["loss"]), iter_id)
  
  info_str = ("iteration {}: loss = {:.3f} accuracy = {:.3f}").format(iter_id, iter_info["loss"], iter_info["accuracy"])
  if iter_info.get("lr", None) is not None:
    info_str = info_str + " lr = {:.5f}".format(iter_info["lr"])
    summary_writer.add_summary(make_summary("iteration/" + "lr", iter_info["lr"]), iter_id)
  return info_str

def update_epoch_info(summary_writer, epoch_info):
  global_step = epoch_info["global_step"]
  summary_writer.add_summary(make_summary("epoch/" + "accuracy", epoch_info["accuracy"]), global_step)
  summary_writer.add_summary(make_summary("epoch/" + "loss", epoch_info["loss"]), global_step)
  summary_writer.add_summary(make_summary("epoch/" + "examples_per_sec", epoch_info["examples_per_sec"]), global_step)
  summary_writer.add_summary(make_summary("epoch/" + "lr", epoch_info["lr"]), global_step)

  info_str = ("-- global_step_val = {} loss = {:.3f} accuracy = {:.3f} "+
              "examples_per_sec = {:.3f}").format(global_step, epoch_info["loss"], 
                          epoch_info["accuracy"], epoch_info["examples_per_sec"])
  return info_str


def make_summary(name, value):
  this_summary = tf.Summary()
  item = this_summary.value.add()
  item.tag = name
  item.simple_value = value
  return this_summary


def train_model(infile):
  """Runs the model on the given data."""
  batch_size = FLAGS.batch_size
  train_dir = FLAGS.train_dir
  starter_learning_rate = FLAGS.start_learning_rate
  
  logging.info("Start loading the data")
  train_data = reader_bow.DataSet(infile)
  logging.info("Finish loading the data")

  epoch_size = int(np.floor(train_data.num_examples/batch_size))

  
  if not os.path.exists(train_dir):
    os.makedirs(train_dir)

  parameter_info = ["Parameter Info:"]
  parameter_info.append("====================")
  parameter_info.append("#train_examples = {}".format(train_data.num_examples))
  parameter_info.append("batch_size = {}\nstarter_learning_rate = {}".
               format(batch_size,starter_learning_rate))
  parameter_info.append("#iterations per epoch = {}".format(epoch_size))
  parameter_info.append("====================")
  logging.info("\n".join(parameter_info))
  
  if not os.path.exists(train_dir):
    os.makedirs(train_dir)
  

  with tf.Graph().as_default(), tf.Session() as session:
    logging.info("Start constructing computation graph.")
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
    i_pl = tf.placeholder(tf.float32, [batch_size, train_data.image_feature_dim])
    bow_pl = tf.placeholder(tf.float32, [batch_size, len(train_data.vocabulary)])
    
    labels_pl = tf.placeholder(tf.int32, [batch_size, train_data.num_classes]) 
    
    # Build a Graph that computes predictions from the inference model.
    predictions = models.build_lr_bow(i_pl, bow_pl, train_data._num_classes)
   
    # Add to the Graph the Ops for loss calculation.
    loss = models.calculate_xentropy_loss(predictions, labels_pl)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = train(loss, global_step, starter_learning_rate)
    lr = tf.get_collection("lr")[0]
    
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()
    
    init = tf.initialize_all_variables()
    
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    
    if FLAGS.retrain: # whether to retrain from a previous model?
      ckpt = tf.train.get_checkpoint_state(train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
        logging.info("**************** Retrain ***************************")
        logging.info("Restored the latest model global_step = {}.".format(ckpt.model_checkpoint_path))
        logging.info("****************************************************")
      else:
        logging.error("No trained model found at {}".format(train_dir))
    else:
      # you need to clear the training directory
      session.run(init)
    
    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(train_dir, session.graph)
    logging.info("Finish constructing computation graph.")

    
    logging.info("Start Training")
    
    iter_eval = myeval.Eval_Metrics() 

    for _ in xrange(int(1e08)):
      
      start_time = time.time()
      
      # get the next mini-batch data
      batch_data = train_data.next_batch(batch_size)
      
      batch_binary_labels = batch_data["labels"].copy()
      batch_binary_labels[batch_binary_labels < 0] = 0
      
      feed_dict = {bow_pl:batch_data["BoWs"], i_pl:batch_data["Is"], labels_pl:batch_binary_labels}
       
      # run the graph
      global_step_val, _, loss_val, predictions_val, lr_val = session.run([global_step, train_op, loss, predictions, lr], feed_dict=feed_dict)
      duration = time.time() - start_time
     
      #logging.info(iter_info_str)
      iter_accuracy = myeval.calculate_accuracy_batch(predictions_val, batch_data["labels"])
      iter_eval.accumulate(iter_accuracy, loss_val, duration, FLAGS.batch_size)
      
      
      if global_step_val % 30 == 0:
        iter_info = iter_eval.get_metrics()
        iter_info["global_step"] = global_step_val
        iter_info["lr"] = float(lr_val)
        epoch_info_str = update_iter_info(summary_writer, iter_info)
        logging.info(epoch_info_str)
        
        summary_str = session.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step_val)
        summary_writer.flush()
      
      if global_step_val % epoch_size == 0:
        # reach an epoch
        epoch_info = iter_eval.get_metrics_and_clear()
        epoch_info["global_step"] = global_step_val
        epoch_info["lr"] = float(lr_val)
        epoch_info_str = update_epoch_info(summary_writer, epoch_info)
        logging.info(epoch_info_str)
        
        # save model
        saver.save(session, os.path.join(train_dir, "{}.model".format(os.path.basename(FLAGS.train_dir))), global_step=global_step_val)

        # Update the events file.
        summary_str = session.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step_val)
        summary_writer.flush()
        
        
def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to the training feature pickple file")
  train_model(FLAGS.data_path)


if __name__ == "__main__":
  tf.app.run()
