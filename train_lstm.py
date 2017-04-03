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

import utils
import eval as myeval
import reader_embedding


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
 
flags = tf.flags
flags.DEFINE_string("train_dir", "/Users/lujiang/run/", "Training output directory")
flags.DEFINE_string("data_path", "/Users/lujiang/data/memex_dataset/exp/lr_embedding_tr.p", "data_path")
flags.DEFINE_string("photo_feat", "/Users/lujiang/data/memex_dataset/exp/photo_feat.p", "photo_feat")
flags.DEFINE_string("model", "lstm_q", "model_name")


flags.DEFINE_integer("batch_size", 64, "training batch size")
flags.DEFINE_boolean("retrain", False, "whether to retrain or not")

FLAGS = flags.FLAGS


def train_model(infile):
  """Runs the model on the given data."""
  batch_size = FLAGS.batch_size
  train_dir = os.path.join(FLAGS.train_dir, FLAGS.model)
  
  logging.info("Start loading the data")
  train_data = reader_embedding.DataSet(infile)
  logging.info("Finish loading the data")

  epoch_size = int(np.floor(train_data.num_examples/batch_size))

  
  if not os.path.exists(train_dir):
    os.makedirs(train_dir)

  parameter_info = ["Parameter Info:"]
  parameter_info.append("====================")
  parameter_info.append("model = {}".format(FLAGS.model))
  parameter_info.append("#train_examples = {}".format(train_data.num_examples))
  parameter_info.append("batch_size = {}".format(batch_size))
  parameter_info.append("#iterations per epoch = {}".format(epoch_size))
  parameter_info.append("====================")
  logging.info("\n".join(parameter_info))
  
  if not os.path.exists(train_dir):
    os.makedirs(train_dir)

  with tf.Graph().as_default(), tf.Session() as session:
    logging.info("Start constructing computation graph.")
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
    
    placeholders = {}
    placeholders["Qs"] = tf.placeholder(tf.int32, [batch_size, train_data.max_window_size])
    placeholders["Qs_l"] = tf.placeholder(tf.int32, [batch_size,])
    
    placeholders["Is"] = tf.placeholder(tf.int32, [batch_size, train_data.max_window_size])
    placeholders["Gs"] = tf.placeholder(tf.int32, [batch_size, train_data.max_window_size])
    placeholders["Ts"] = tf.placeholder(tf.int32, [batch_size, train_data.max_window_size])
    placeholders["labels"] = tf.placeholder(tf.int32, [batch_size, train_data.num_classes])

    # Build a Graph that computes predictions from the inference model.
    predictions = getattr(models, "build_{}".format(FLAGS.model))(placeholders, FLAGS.photo_feat, len(train_data.vocabulary), train_data._num_classes)

    # Add to the Graph the Ops for loss calculation.
    loss = models.calculate_softmax_loss(predictions, placeholders["labels"])

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = getattr(models, "train_{}".format(FLAGS.model))(loss, global_step)
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

    for _ in xrange(int(1e06)):
      
      start_time = time.time()
      
      # get the next mini-batch data
      batch_data = train_data.next_batch(batch_size)
      
      batch_binary_labels = batch_data["labels"].copy()
      batch_binary_labels[batch_binary_labels < 0] = 0
      
      feed_dict = {}
      modalities = [t for t in train_data._modalities if t not in ["qids", "labels", "As", "PTs", "ATs"]]
      for k in modalities:
        feed_dict[placeholders[k]] = batch_data[k]
      
      feed_dict[placeholders["labels"]] = batch_binary_labels
       
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
        epoch_info_str = utils.update_iter_info(summary_writer, iter_info)
        logging.info(epoch_info_str)
        
        summary_str = session.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step_val)
        summary_writer.flush()
      
      if global_step_val % epoch_size == 0:
        # reach an epoch
        epoch_info = iter_eval.get_metrics_and_clear()
        epoch_info["global_step"] = global_step_val
        epoch_info["lr"] = float(lr_val)
        epoch_info_str = utils.update_epoch_info(summary_writer, epoch_info)
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
