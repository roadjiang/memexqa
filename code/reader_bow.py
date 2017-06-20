'''
Created on Feb 3, 2017

@author: lujiang
'''


import numpy as np
import pickle


class DataSet(object):
  


  def __init__(self, infile):
    dataset = pickle.load(open(infile, "rb"))
    
    self._data = dataset
    
    self._modalities = ["qids", "labels", "As", "BoWs", "Is" ]

        
#     self._data["qids"] = dataset["qids"]
#     self._labels = dataset["labels"]
#     self._Is = dataset["Is"]
#     self._BoWs = dataset["BoWs"]
#     self._vocabulary = tr["vocabulary"]
#     self._classes = tr["classes"]
    
    # pad and truncate input questions
    self._image_feat_dim = self._data["Is"].shape[1]

    self._num_examples = len(self._data["qids"])
    self._vocabulary_size = len(self._data["vocabulary"])
    self._num_classes = len(self._data["classes"])
    
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def num_examples(self):
    return self._num_examples
  
  @property
  def num_classes(self):
    return self._num_classes
  
  @property
  def vocabulary(self):
    return self._data["vocabulary"]

  @property
  def classes(self):
    return self._data["classes"]
  
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
      
      for k in self._modalities:
        if k in self._data:
          self._data[k] = self._data[k][perm]
      
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self.num_examples
    end = self._index_in_epoch
    
    batch = {}
    for k in self._modalities:
      if k in self._data:
        if k == "As": continue
        batch[k] = self._data[k][start:end]
    
    batch["qids"] = batch["qids"].reshape(-1,1)


    labels = np.zeros((batch_size, self.num_classes))
    for i in range(batch_size):
      labels[i,self._data["As"][start+i,]] = -1
      labels[i,self._data["labels"][(start+i)]] = 1
    
    batch["labels"] = labels
    
    return batch
  

if __name__ == '__main__':
  train = DataSet("/Users/lujiang/data/memex_dataset/exp/bow_ts.p")
  print train.num_examples
  print train.num_classes
  for _ in range(20):
    batch = train.next_batch(100)
      