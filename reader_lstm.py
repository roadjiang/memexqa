'''
Created on Feb 3, 2017

@author: lujiang
'''


import numpy as np
import pickle


class DataSet(object):
  


  def __init__(self, infile, max_input_window_size = 8):
    tr = pickle.load(open(infile, "rb"))
    
    self._ids = tr["qids"]
    self._labels = tr["labels"]
    self._Qs = tr["Qs"]
    self._Is = tr["Is"]
    self._As = tr["As"]
    self._Gs = tr["Gs"]
    self._Ts = tr["Ts"]
    self._vocabulary = tr["vocabulary"]
    self._classes = tr["classes"]
    
    # pad and truncate input questions
    self._end_token = "$"
    self._end_token_id = np.where(self._vocabulary == "$")[0][0]
    self._max_window_size = max_input_window_size
    self._image_feat_dim = self._Is.shape[1]

    self._Qs = self._pad_input(self._Qs, max_input_window_size, self._end_token_id)
    self._Ts = self._pad_input(self._Ts, max_input_window_size, self._end_token_id)
    self._Gs = self._pad_input(self._Gs, max_input_window_size, self._end_token_id)

    
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
  
  def _pad_input(self, Qs, max_len, end_token_id):
    result = np.zeros([len(Qs), max_len], dtype=np.int32)
    result.fill(end_token_id)
    for i in range(len(Qs)):
      if len(Qs[i]) < max_len:
        result[i,0:len(Qs[i])] = Qs[i]
      else:
        result[i,0:max_len] = Qs[i][0:max_len]
    return result

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
      self._Qs = self._Qs[perm]
      self._As = self._As[perm]
      self._Is = self._Is[perm]
      self._Ts = self._Ts[perm]
      self._Gs = self._Gs[perm]
      self._labels = self._labels[perm]
      self._ids = self._ids[perm]
      
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self.num_examples
    end = self._index_in_epoch
    
    ids = self._ids[start:end].reshape(-1,1)
    Qs = self._Qs[start:end]
    Is = self._Is[start:end]
    Gs = self._Gs[start:end]
    Ts = self._Ts[start:end]    
    labels = np.zeros((batch_size, self.num_classes))
    
    for i in range(batch_size):
      labels[i,self._As[start+i,]] = -1
      labels[i,self._labels[(start+i)]] = 1
    
    batch = {"Qs":Qs, "labels":labels, "Is":Is, "ids":ids, "Gs":Gs, "Ts":Ts}
    return batch
  
  def _debug_verify(self, Q, label, qid):
    id2word = {i:self.vocabulary[i] for i in range(len(self.vocabulary))}
    id2class = {i:self.classes[i] for i in range(len(self.classes))}
    print qid[0], " ".join([id2word[t] for t in Q if t != self._end_token_id]) + "?"
    print id2class[np.where(label == 1)[0][0]], [id2class[t] for t in np.where(label == -1)[0]]

if __name__ == '__main__':
  train = DataSet("/Users/lujiang/data/memex_dataset/exp/lr_embedding_ts.p")
  print train.num_examples
  print train.num_classes
  for _ in range(2000):
    batch = train.next_batch(100)
    train._debug_verify(batch["Qs"][4], batch["labels"][4], batch["ids"][4])
      