'''
Created on Feb 3, 2017

@author: lujiang
'''


import numpy as np
import pickle


class DataSet(object):
  


  def __init__(self, infile):
    tr = pickle.load(open(infile, "rb"))
    
    self._ids = tr["qids"]
    self._labels = tr["labels"]
    self._Is = tr["Is"]
    self._As = tr["As"]
    self._BoWs = tr["BoWs"]
    self._vocabulary = tr["vocabulary"]
    self._classes = tr["classes"]
    
    # pad and truncate input questions
    self._image_feat_dim = self._Is.shape[1]

    self._num_examples = len(self._ids)
    self._vocabulary_size = len(self._vocabulary)
    self._num_classes = len(self._classes)
     
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def num_examples(self):
    return self._num_examples
  
  @property
  def max_window_size(self):
    return self._max_window_size
  
  @property
  def num_classes(self):
    return self._num_classes
  
  @property
  def vocabulary(self):
    return self._vocabulary

  @property
  def classes(self):
    return self._classes
  
  @property
  def epochs_completed(self):
    return self._epochs_completed;
  
  @property
  def image_feature_dim(self):
    return self._image_feat_dim
  

  def next_batch(self, batch_size):
    """ store everything in memory for small-scale application"""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    
    if self._index_in_epoch > self._num_examples:
      # finished epoch
      self._epochs_completed += 1
      # shuffle the data
      perm = np.arange(self.num_examples)
      np.random.shuffle(perm)
      self._As = self._As[perm]
      self._Is = self._Is[perm]
      self._BoWs = self._BoWs[perm]
      self._labels = self._labels[perm]
      self._ids = self._ids[perm]
      
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self.num_examples
    end = self._index_in_epoch
    
    ids = self._ids[start:end].reshape(-1,1)
    Is = self._Is[start:end]
    BoWs = self._BoWs[start:end]

    labels = np.zeros((batch_size, self.num_classes))
    for i in range(batch_size):
      labels[i,self._As[start+i,]] = -1
      labels[i,self._labels[(start+i)]] = 1
    
    batch = {"labels":labels, "Is":Is, "ids":ids, "BoWs":BoWs}
    return batch
  

if __name__ == '__main__':
  train = DataSet("/Users/lujiang/data/memex_dataset/exp/bow_ts.p")
  print train.num_examples
  print train.num_classes
  for _ in range(20):
    batch = train.next_batch(100)
      