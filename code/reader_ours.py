'''
Created on Feb 3, 2017

@author: lujiang
'''


import numpy as np
import pickle
import utils


class DataSet(object):
  


  def __init__(self, infile, topn_retrived = 2, max_input_window_size = 8):
    dataset = pickle.load(open(infile, "rb"))
    
    self._data = dataset

    # pad and truncate input questions
    self._end_token = "$"
    self._end_token_id = np.where(self._data["vocabulary"] == "$")[0][0]
    self._max_window_size = max_input_window_size
    
    self._data["topn"] = topn_retrived
    self._data["Gs"] = np.array([t[0:topn_retrived]for t in self._data["Gs"]])
    
    self._data["Ts"] = np.array([t[0:topn_retrived]for t in self._data["Ts"]])
    self._data["PTs"] = np.array([t[0:topn_retrived]for t in self._data["PTs"]])
    
    self._data["Qs"], self._data["Qs_l"] = utils.pad_input(self._data["Qs"], 12, self._end_token_id)
    #self._data["Is"], self._data["Is_l"] = utils.pad_input(self._data["Is"], 5, self._end_token_id) 

    #self._data["Ts"], self._data["Ts_l"] = utils.pad_input(self._data["Ts"], max_input_window_size, self._end_token_id)
    #self._data["Gs"], self._data["Gs_l"] = utils.pad_input(self._data["Gs"], max_input_window_size, self._end_token_id)

    #self._data["ATs"], self._data["ATs_l"] = utils.pad_input(self._data["ATs"], 16, self._end_token_id)
    #self._data["PTs"], self._data["PTs_l"] = utils.pad_input(self._data["PTs"], 16, self._end_token_id)
    

    self._modalities = ["qids", "labels", "As", "Qs", "Is", "Ts", "Gs", "PTs", "Qs_l"]

    self._num_examples = len(self._data["qids"])
    self._vocabulary_size = len(self._data["vocabulary"])
    self._num_classes = len(self._data["classes"])
    
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
    return self._data["vocabulary"]

  @property
  def classes(self):
    return self._data["classes"]
  
  @property
  def topn(self):
    return self._data["topn"]
  
  @property
  def epochs_completed(self):
    return self._epochs_completed;


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
        if k in ["ATs"]: continue
        batch[k] = self._data[k][start:end]
        
    batch["qids"] = batch["qids"].reshape(-1,1)
          
    labels = np.zeros((batch_size, self.num_classes))
    for i in range(batch_size):
      labels[i,self._data["As"][start+i,]] = -1
      labels[i,self._data["labels"][(start+i)]] = 1
    
    batch["labels"] = labels
    return batch
  
  def _debug_verify(self, batch):
    id2word = {i:self.vocabulary[i] for i in range(len(self.vocabulary))}
    id2class = {i:self.classes[i] for i in range(len(self.classes))}
    current_id = np.random.choice(len(batch),1)[0]
    print batch["qids"][current_id], " ".join([id2word[t] for t in batch["Qs"][current_id] if t != self._end_token_id]) + "?"
    print id2class[np.where(batch["labels"][current_id] == 1)[0][0]], [id2class[t] for t in np.where(batch["labels"][current_id] == -1)[0]]
    #print np.where(batch["labels"][current_id] == 1)[0]
    #print batch["Is"][current_id]
    print np.where(batch["labels"][current_id] == 1)[0][0]
    print batch["PTs"][current_id]
 
if __name__ == '__main__':
  train = DataSet("/Users/lujiang/data/memex_dataset/exp/ours_ts.p")
  print train.num_examples
  print train.num_classes
  for _ in range(2000):
    batch = train.next_batch(100)
    train._debug_verify(batch)
      