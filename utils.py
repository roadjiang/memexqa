'''
Created on Mar 31, 2017

@author: lujiang
'''
import tensorflow as tf
import numpy as np

def flat_arrays(data):
  result = []
  for t in data: 
    result.append(sum(t,[]))
  return result

def pad_input(data, max_len, end_token_id):
  result = np.zeros([len(data), max_len], dtype=np.int32)
  result_len = np.zeros(len(data), dtype=np.int32)
  result.fill(end_token_id)
  for i in range(len(data)):
    if len(data[i]) <= max_len:
      result[i,0:len(data[i])] = data[i]
      result_len[i] = len(data[i])
    else:
      result[i,0:max_len] = data[i][0:max_len]
      result_len[i] = max_len
  return result, result_len


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