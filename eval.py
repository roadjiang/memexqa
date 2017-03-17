'''
Created on Mar 9, 2017

@author: lujiang
'''
import numpy as np
import pickle
import sys
import csv

def load_csv(input_file, header=False):
  csv.field_size_limit(sys.maxsize)
  intable = []
  with open(input_file, 'rb') as csvfile:
    myreader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in myreader:
      intable.append(row)
  return intable


def evaluate_csv(pred_csv_path, qa_pickle_path):
  qa, _ = pickle.load(open(qa_pickle_path, "rb"))
  qa_dict = {t["question_id"]:t for t in qa}
  
  answer_sheet = load_csv(pred_csv_path)
  
  scores = []
  qtypes = []
  
  for t in answer_sheet:
    qid = int(t[0])
    qa_item = qa_dict[qid]
    qtypes.append(qa_item["question"].split(" ")[0].lower())
    if qa_item["ml_answer_class"] == int(t[2]):
      scores.append(1)
    else:
      scores.append(0)

  scores = np.array(scores)
  qtypes = np.array(qtypes)
  
  print "Total #QA: {}".format(len(scores))
  
  unique_qtype = np.unique(qtypes)
  qtype_acc = []
  for t in unique_qtype:
    qtype_acc.append("{:.3f}".format(np.mean(scores[np.where(qtypes==t)])))
    
  unique_qtype = np.append(unique_qtype, "overall")
  qtype_acc.append("{:.3f}".format(np.mean(scores)))
  
  print " ".join(unique_qtype.tolist())
  print " ".join(qtype_acc)
  

def calculate_accuracy_batch(predictions, labels):
  accuracy = 0.0
  for i in range(labels.shape[0]):
    candidate_class_ids = np.where(labels[i,] != 0)[0]
    pred_class_id = candidate_class_ids[np.argmax(predictions[i,candidate_class_ids])]
    if labels[i, pred_class_id] > 0:  accuracy += 1
  
  accuracy = accuracy / labels.shape[0]
  return accuracy

def get_prediction_batch(predictions, labels):
  result = []
  for i in range(labels.shape[0]):
    candidate_class_ids = np.where(labels[i,] != 0)[0]
    pred_class_id = candidate_class_ids[np.argmax(predictions[i,candidate_class_ids])]
    result.append(pred_class_id)
  return result

class Eval_Metrics(object):
  
  def __init__(self):
    self.clear()

  def accumulate(self, iter_accuracy, iter_loss, iter_duration, batch_size):
    self._accuracy_sum = self._accuracy_sum + iter_accuracy * batch_size
    self._loss_sum = self._loss_sum + iter_loss * batch_size
    self._runtime_sum = self._runtime_sum + iter_duration * batch_size
    self._num_seen_examples = self._num_seen_examples + batch_size

  def get_metrics(self):
    info_dict = {}
    info_dict["examples_per_sec"] = self._num_seen_examples/self._runtime_sum
    info_dict["accuracy"] = self._accuracy_sum/self._num_seen_examples
    info_dict["loss"] = self._loss_sum/self._num_seen_examples
    return info_dict
  
  def get_metrics_and_clear(self):
    result = self.get_metrics()
    self.clear()
    return result
  
  def clear(self):
    self._runtime_sum = 0.0
    self._accuracy_sum = 0.0
    self._loss_sum = 0.0
    self._num_seen_examples = 0
  

if __name__ == '__main__':
  evaluate_csv("/Users/lujiang/run/lr_embedding_ts/predictions.csv", 
               "/Users/lujiang/data/memex_dataset/exp/qa_album.p")