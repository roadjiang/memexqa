'''
Created on Mar 10, 2017

@author: lujiang
'''

import time
import os


import utils

import numpy as np
import models
import tensorflow as tf
import logging

import eval as myeval
import reader_embedding

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

flags = tf.flags

flags.DEFINE_string("train_dir", "/Users/lujiang/run/", "Training output directory")
flags.DEFINE_string("test_dir", "/Users/lujiang/run/", "Testing output directory")
flags.DEFINE_string("model", "lr_embedding_q_i", "model_name")

flags.DEFINE_string("data_path", "/Users/lujiang/data/memex_dataset/exp/lr_embedding_ts.p", "data_path")
flags.DEFINE_string("photo_feat", "/Users/lujiang/data/memex_dataset/exp/photo_feat.p", "photo_feat")
flags.DEFINE_string("ground_truth_file", "/Users/lujiang/data/memex_dataset/exp/qa_album.p", "ground_truth_file")

flags.DEFINE_integer("batch_size", 4, "test batch size")

FLAGS = flags.FLAGS

def test_model(infile):
  """Runs the trained model"""
  batch_size = FLAGS.batch_size
  train_dir = os.path.join(FLAGS.train_dir, FLAGS.model)
  test_dir = os.path.join(FLAGS.test_dir, FLAGS.model)
  
  logging.info("Start loading the data")
  test_data = reader_embedding.DataSet(infile)
  logging.info("Finish loading the data")

  epoch_size = int(np.floor(test_data.num_examples/batch_size)) + 1


  if not os.path.exists(test_dir):
    os.makedirs(test_dir)
  
  parameter_info = ["Parameter Info:"]
  parameter_info.append("====================")
  parameter_info.append("model = {}".format(FLAGS.model))
  parameter_info.append("loaded trained model: {}".format(train_dir))
  parameter_info.append("#test_examples = {}".format(test_data.num_examples))
  parameter_info.append("test_batch_size = {}".format(batch_size))
  parameter_info.append("#iterations per epoch = {}".format(epoch_size))

  parameter_info.append("====================")
  logging.info("\n".join(parameter_info))
  
  
  logging.info("Start training.")
  

  with tf.Graph().as_default(), tf.Session() as session:
    logging.info("Start constructing computation graph.")
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)


    placeholders = {}
    placeholders["Qs"] = tf.placeholder(tf.int32, [batch_size, test_data.max_window_size])
    placeholders["Is"] = tf.placeholder(tf.int32, [batch_size, test_data.max_window_size])
    placeholders["Gs"] = tf.placeholder(tf.int32, [batch_size, test_data.max_window_size])
    placeholders["Ts"] = tf.placeholder(tf.int32, [batch_size, test_data.max_window_size])
    placeholders["labels"] = tf.placeholder(tf.int32, [batch_size, test_data.num_classes])

    placeholders["ATs"] = tf.placeholder(tf.int32, [batch_size, test_data.max_window_size])
    placeholders["PTs"] = tf.placeholder(tf.int32, [batch_size, test_data.max_window_size])

    
    # Build a Graph that computes predictions from the inference model.
    predictions = getattr(models, "build_{}".format(FLAGS.model))(placeholders, FLAGS.photo_feat, len(test_data.vocabulary), test_data._num_classes)

    # Add to the Graph the Ops for loss calculation.
    loss = models.calculate_softmax_loss(predictions, placeholders["labels"])

    saver = tf.train.Saver()
    
    ckpt = tf.train.get_checkpoint_state(train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(session, ckpt.model_checkpoint_path)
      logging.info("Restored the latest model global_step = {}.".format(ckpt.model_checkpoint_path))
    else:
      logging.error("No trained model found at {}!".format(train_dir))
    
    logging.info("Finish constructing computation graph.")


    assert test_data.epochs_completed == 0, "Must start from epoch 0."
    
    epoch_eval =  myeval.Eval_Metrics()
    class_vocabulary = test_data.classes

    global_step_val = 0
    results = []
    
    start_time = time.time()
    
    for _ in xrange(int(1e10)):
    
      # get the next mini-batch data
      batch_data = test_data.next_batch(batch_size)
      
      batch_binary_labels = batch_data["labels"].copy()
      batch_binary_labels[batch_binary_labels < 0] = 0
      
      feed_dict = {}
      modalities = [t for t in test_data._modalities if t not in ["qids", "labels", "As", "PTs", "ATs", "Qs_l", "Is_l"]]
      for k in modalities:
        feed_dict[placeholders[k]] = batch_data[k]
      
      feed_dict[placeholders["labels"]] = batch_binary_labels
            
      global_step_val, loss_val, predictions_val = session.run([global_step, loss, predictions], feed_dict=feed_dict)
      global_step_val = global_step_val + 1
      
      #predictions_val = np.random.rand(batch_size, test_data._num_classes) # random baseline
      duration = time.time() - start_time

      iter_accuracy = myeval.calculate_accuracy_batch(predictions_val, batch_data["labels"])
      iter_pred_classes = myeval.get_prediction_batch(predictions_val, batch_data["labels"])
      epoch_eval.accumulate(iter_accuracy, loss_val, duration, FLAGS.batch_size)
      
      
      # add predictions the results
      for i in range(batch_data["qids"].shape[0]):
        true_class = np.where(batch_binary_labels[i]>0)[0][0]
        pred_class = iter_pred_classes[i]
        results.append([int(batch_data["qids"][i]), true_class, pred_class, 
                       class_vocabulary[true_class].encode('utf8'), class_vocabulary[pred_class].encode('utf8')])

      if test_data.epochs_completed == 1: break # read rhe end of an epoch

    end_time = time.time()
    logging.info("== reach the end of an epoch using {}".format(end_time - start_time))

    epoch_info = epoch_eval.get_metrics_and_clear()
    epoch_info_str = ("-- global_step_val = {} loss = {:.3f} accuracy = {:.3f} "+
              "examples_per_sec = {:.3f}").format(global_step_val, epoch_info["loss"], 
                          epoch_info["accuracy"], epoch_info["examples_per_sec"])
    logging.info(epoch_info_str)
    
    # output the predictions
    utils.write_prediction_csv(os.path.join(test_dir, "predictions.csv"), results)
    myeval.evaluate_csv(os.path.join(test_dir, "predictions.csv"), FLAGS.ground_truth_file)



def main(_):
  if not FLAGS.data_path: raise ValueError("Must set --data_path to the test data file")
  test_model(FLAGS.data_path)
  #test_call_once(np.array([[1,1]]))
  
if __name__ == '__main__':
  tf.app.run()