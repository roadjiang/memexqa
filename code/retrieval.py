'''
Created on May 1, 2017

@author: lujiang
'''

import re
import gensim
import numpy as np

from nltk.corpus import stopwords
stopword_dict = {stopword:"indexed" for stopword in stopwords.words('english')}

SYMBOLS = "[()\\[\\]\\*\\^\\.\\-/\":;,&!<>{}_%\\?~$@`#\t\r\n']"
import nltk
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

def parse_questions(instring):
  """ parse question words to index them.
  """
  instring = re.sub(SYMBOLS, " ", instring).lower()
  result = []
  for this_word in nltk.word_tokenize(instring):
    if this_word  in stopword_dict: continue
    if re.match("\\d+", this_word): continue
    result.append(this_word)
  return result



def generate_concept_exact_query(qa, concept_dict):
  concept_name_to_id = {}
  for t in concept_dict:
    concept_name_to_id[t[1]] = t[0]
  
  queries = []
  for t in qa:
    this_query = []
    tokens = parse_questions(t["question"])
    for k in tokens:
      if k in concept_name_to_id: this_query.append(concept_name_to_id[k])
    queries.append(this_query)
  
  with open("data/memex_dataset/retrieval/concepts_exact.queries", 'wb') as myfile:
    for t in queries:
      if not t: myfile.write("none\n")
      else: myfile.write(",".join(t)+"\n")


def get_query_scope(pids, qa, album):
  aid2photo_row_id = {t["album_id"]: [np.where(pids==k)[0][0] for k in t["photo_ids"]] for t in album}
  scopes = []
  for t in qa:
    this_scope = []
    for a in t["album_ids"]:
      this_scope.extend(aid2photo_row_id[a])
    scopes.append(this_scope)
  
  with open("data/memex_dataset/retrieval/search_scope.txt", 'wb') as myfile:
    for t in scopes:
      myfile.write(",".join([str(k) for k in t])+"\n")



def get_ground_truth(pids, qa):
  truths = []
  pid2photo_row_id = {pids[i]:i for i in xrange(len(pids))}
  for t in qa:
    this_truth = [pid2photo_row_id[k] for k in t["evidence_photo_ids"]]
    truths.append(this_truth)
  
  with open("data/memex_dataset/retrieval/ground_truth.txt", 'wb') as myfile:
    for t in truths:
      myfile.write(",".join([str(k) for k in t])+"\n")
  
      
def generate_concept_embedding_query(qa, concept_dict):
  concept_names = np.array([t[1] for t in concept_dict])
  concept_ids = np.array([t[0] for t in concept_dict])
  
  embedding = gensim.models.Word2Vec.load_word2vec_format("/Users/lujiang/data/word2vec/GoogleNews-vectors-negative300.bin.gz", binary =True)

  queries = []
  for t in qa:
    this_query = []
    query_words = parse_questions(t["question"])
    this_query.append(_embedding_helper(embedding, query_words, concept_names, concept_ids))
    queries.append(this_query[0])
  
  with open("data/memex_dataset/retrieval/concepts_embedding.queries", 'wb') as myfile:
    for t in queries:
      if not t: myfile.write("none\n")
      else: myfile.write(",".join(t)+"\n")

def _embedding_helper(embedding, query_words, concept_names, concept_ids):
  sim_vec = np.zeros(len(concept_names))
  for i in xrange(len(concept_names)):
    if concept_names[i] not in embedding: continue
    sim_value = 0.0
    for query_word in query_words:
      if query_word in embedding: 
          sim_value = sim_value + embedding.similarity(concept_names[i] , query_word)
    sim_vec[i] = sim_value

  return concept_ids[sim_vec.argsort()[::-1][:5]]

