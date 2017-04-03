'''
Created on Mar 9, 2017

@author: lujiang
'''

import numpy as np
import pickle
import csv
import nltk

import collections
import re
import json
import os

DATES = "on ([JFMAJSOND][^ ]+? \d+ \d+)"
SYMBOLS = "[()\\[\\]\\*\\^\\.\\-/\":;,&!<>{}_%\\?~$@`#\t\r\n']+"

from nltk.corpus import stopwords
stopword_dict = {stopword:"indexed" for stopword in stopwords.words('english')}

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

from nltk.corpus import wordnet


def get_wordnet_pos(treebank_tag):
  if treebank_tag.startswith("J"):
    return wordnet.ADJ
  elif treebank_tag.startswith("V"):
    return wordnet.VERB
  elif treebank_tag.startswith("N"):
    return wordnet.NOUN
  elif treebank_tag.startswith("R"):
    return wordnet.ADV
  else:
    return wordnet.NOUN

def parse_answer(inanswer):
  """Do not remove stop words. Every word is useful"""
  instring = re.sub(SYMBOLS, " ", inanswer).lower()
  result = []
  tokens = nltk.pos_tag(nltk.word_tokenize(instring))
  for i in range(len(tokens)):
    token = tokens[i]
    if i == 0 and (token[0] == "a" or token[0] == "the" or token[0] == "an"): continue
    this_word = wordnet_lemmatizer.lemmatize(token[0], get_wordnet_pos(token[1]))
    result.append(this_word)
  return "_".join(result)

def parse_album_date(indate):
  return indate.lower().replace("on", "").strip().replace(" ", "_")

def get_album_place_tokens(inplace):
  if inplace is None: return []
  tokens = inplace.lower().split(",")
  return [t.strip().replace(" ","_") for t in tokens]
  

def get_album_info_tokens(ininfo):
  """album title and description"""
  instring = re.sub(SYMBOLS, " ", ininfo).strip().lower()
  result = []
  for token in nltk.pos_tag(nltk.word_tokenize(instring)):
    this_word = wordnet_lemmatizer.lemmatize(token[0], get_wordnet_pos(token[1]))
    if this_word not in stopword_dict:
      result.append(this_word)
  return result
  
def get_photo_title_tokens(intitle):
  return get_album_info_tokens(intitle)
  

def get_question_tokens(inquestion):
  """Do not remove stop words. Every word is useful"""
  # make date a single word
  date_matches = re.findall(DATES, inquestion)
  for t in date_matches:  inquestion = inquestion.replace(t, t.replace(" ", "_"))
  
  instring = re.sub("[\? \t]+", " ", inquestion).lower().strip()
  result = []
  for token in nltk.pos_tag(nltk.word_tokenize(instring)):
    this_word = wordnet_lemmatizer.lemmatize(token[0], get_wordnet_pos(token[1]))
    result.append(this_word)
  return result


def load_txt(input_file):
  intable = []
  with open(input_file, 'rb') as csvfile:
    myreader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in myreader:
      if len(row) == 1:
        intable.append(row[0])
      else:
        intable.append(row)
  return intable

def load_npz(featfile, pids):
  tmp = np.load(featfile)
  m = np.zeros([len(pids),len(tmp[tmp.keys()[0]])])
  for i in range(len(pids)):
    m[i,] = tmp[pids[i]]
  return m

# def text_retrival(question):
#   intokens = parse_questions(inquestion, topk=2)
#   q_vec = lil_matrix((len(dictionary),1), dtype=np.float32)
#   N = sparse_matrix.shape[0]
#   
#   for token in intokens:
#     tid = dictionary.get(token, None)
#     if tid: q_vec[tid,0] += (1.0/len(intokens)) * np.log(N/(df[tid]+1))
#   
#   cur_sim = np.dot(sparse_matrix, q_vec)
#   cur_sim = np.squeeze(cur_sim.toarray())
#   ids = np.argsort(-1*cur_sim, axis=0)[0:topk]
#   return ids
  
  


def build_dataset():
  idfile = "/Users/lujiang/data/memex_dataset/ids/shown_photo.ids"
  qafile = "/Users/lujiang/data/memex_dataset/qa.json"
  albumfile = "/Users/lujiang/data/memex_dataset/album_info.json"
  featfile = "/Users/lujiang/data/memex_dataset/resnet_pool5prob.npz"
  #testuserfile = "/Users/lujiang/data/memex_dataset/ids/test_users.split"
  
  badqidlistfile = "/Users/lujiang/data/memex_dataset/ids/bad_question.ids"
  
  END_TOKEN = "$"
  
  with open(albumfile) as infile: album = json.load(infile)
  pids = np.array(load_txt(idfile))
  
  
  with open(qafile) as myfile:  qa = json.load(myfile)  
  bad_qids = np.array(load_txt(badqidlistfile))
  qids = np.unique([t["question_id"] for t in qa])
  qids= np.setdiff1d(qids, bad_qids)
  qa = [t for t in qa if t["question_id"] in qids]

# split by users
#   test_user_dict = {t:t for t in load_txt(testuserfile)}
#   tr_qids = np.array([t["question_id"] for t in qa if t["question_id"] in qids and t["flickr_user_id"] not in test_user_dict])
#   ts_qids = np.array([t["question_id"] for t in qa if t["question_id"] in qids and t["flickr_user_id"] in test_user_dict])
   
  # split by random question
  tr_qids = np.random.choice(qids, int(len(qids)*0.8), replace=False)
  ts_qids = np.setdiff1d(qids,tr_qids)
  print "#qids={}, #tr_qids={}, #ts_qids={}, #bad_qids={}".format(len(qids), len(tr_qids), len(ts_qids), len(bad_qids))

  
  
  #aid2pid = {t["album_id"]:t["photo_ids"] for t in album}

  photo_feat = load_npz(featfile, pids)
  # l2 norm cnn
  for i in range(photo_feat.shape[0]):  photo_feat[i,] = photo_feat[i,]/np.sqrt(np.sum(photo_feat[i,]*photo_feat[i,]))

  
#   aids = np.array(aid2rowid.keys())
#   album_feat = np.zeros([len(aids), photo_feat.shape[1]])
#   for i in range(len(aids)):
#     album_feat[i,] = photo_feat[aid2rowid[aids[i]],].mean(0)
  
  
  correct_choices = [parse_answer(t["answer"]) for t in qa]
  candidate_choices = []
  for t in qa:
    for k in t["multiple_choices_4"]:
      candidate_choices.append(parse_answer(k))
  
  counter = collections.Counter(candidate_choices)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  sampled_candidate_choices = [t[0] for t in count_pairs if t[1] >= 3]
  #len([t for t in count_pairs if t[1]>1])
  
  
  counter = collections.Counter(correct_choices)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  print "class has 1+ sample".format(len([t for t in count_pairs if t[1]>1])*1.0/len(count_pairs))
  
  classes = np.unique(correct_choices + sampled_candidate_choices)
  classes2id_dict = {classes[i]:i for i in range(len(classes))}

  multiple_choice_class_ids = []
  default_choice_class_id = len(classes)-1
  class_hit_ratio = 0.0
  for t in qa: 
    this_class_ids = []
    for k in t["multiple_choices_4"]:
      class_id = classes2id_dict.get(parse_answer(k), None)
      if class_id is not None:
        this_class_ids.append(class_id)
        class_hit_ratio += 1
      else:
        this_class_ids.append(default_choice_class_id)
    multiple_choice_class_ids.append(this_class_ids)
  print "#class={}, hit {}% multiple choices".format(len(classes), class_hit_ratio/len(qa)/4*100)

  
  
  vocabulary = [END_TOKEN]
  for t in qa: vocabulary.extend(get_question_tokens(t["question"]))
  for t in album: 
    vocabulary.append(parse_album_date(t["album_when"]))
    vocabulary.extend(get_album_place_tokens(t["album_where"]))
    vocabulary.extend(get_album_info_tokens(t["album_title"]))
    vocabulary.extend(get_album_info_tokens(t["album_description"]))
    for k in t["photo_titles"]:   vocabulary.extend(get_photo_title_tokens(k))
  vocabulary = np.unique(vocabulary)

  #counter = collections.Counter(vocabulary)
  #count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  vocabulary2id_dict = {vocabulary[i]:i for i in range(len(vocabulary))}
  aid2album = {t["album_id"]:t for t in album}
  
  for i in range(len(qa)):
    t = qa[i]
    t["ml_question"] = [vocabulary2id_dict[k] for k in get_question_tokens(t["question"])]
    t["ml_parsed_answer_text"] = parse_answer(t["answer"])
    t["ml_answer_class"] = classes2id_dict[t["ml_parsed_answer_text"]]
    t["ml_multiple_choice_classes"] = multiple_choice_class_ids[i]
  
  for t in album:
    t["ml_album_when"] = vocabulary2id_dict[parse_album_date(t["album_when"])]
    t["ml_album_where"] = [vocabulary2id_dict[q] for q in get_album_place_tokens(t["album_where"])]
    t["ml_album_title"] = [vocabulary2id_dict[q] for q in get_album_info_tokens(t["album_title"])]
    t["ml_album_description"] = [vocabulary2id_dict[q] for q in get_album_info_tokens(t["album_description"])]
    t["ml_photo_titles"] = []
    for k in t["photo_titles"]:  
      t["ml_photo_titles"].append([vocabulary2id_dict[q] for q in get_photo_title_tokens(k)])

#   album_dict = {t["album_id"]:t for t in album}
#   
#   for t in qa:
#     stacked_when = [album_dict[k]["ml_album_when"] for k in t["album_ids"]]
#     pad_token(stacked_when, MAX_ABLUM_FOR_USER, END_TOKEN)
#     
#     stacked_where = []
#     for k in t["album_ids"]:
#       if len(album_dict[k]["ml_album_where"]) > 0:
#         stacked_where.append(album_dict[k]["ml_album_where"])
#       else:
#         stacked_where.append(END_TOKEN)
#     print pad_token(stacked_where, MAX_ABLUM_FOR_USER, END_TOKEN)

  outdir = "/Users/lujiang/data/memex_dataset/exp/"
  pickle.dump([qa, album], open(os.path.join(outdir, "qa_album.p"), "wb"))
  pickle.dump([pids, photo_feat], open(os.path.join(outdir, "photo_feat.p"), "wb"))




  # embedding data
  modalities = ["qids", "labels", "Is", "Qs", "As", "Ts", "Gs", "PTs", "ATs" ]
  tr = {t:[] for t in modalities}
  ts = {t:[] for t in modalities}
  
  aid2photo_feat_rowid = {t["album_id"]: [np.where(pids==k)[0][0] for k in t["photo_ids"]] for t in album}
  
  np.random.seed(0)
  for t in qa:
    Ts = [aid2album[k]["ml_album_when"] for k in t["album_ids"]]
    Gs = [aid2album[k]["ml_album_where"] for k in t["album_ids"]]
    ATs = [aid2album[k]["ml_album_title"] for k in t["album_ids"]]
    PTs = [sum(aid2album[k]["ml_photo_titles"],[]) for k in t["album_ids"]]
    
    Is = sum([aid2photo_feat_rowid[k] for k in t["album_ids"]],[])
    Is = np.random.choice(Is, min(8, len(Is)), replace=False)
    

    if t["question_id"] in tr_qids:
      tr["qids"].append(t["question_id"])
      tr["labels"].append(t["ml_answer_class"])
      tr["Qs"].append(t["ml_question"])
      tr["As"].append(t["ml_multiple_choice_classes"])
      tr["Is"].append(Is)
      tr["Ts"].append(Ts)
      tr["Gs"].append(Gs)
      tr["ATs"].append(ATs)
      tr["PTs"].append(PTs)
    elif t["question_id"] in ts_qids:
      ts["qids"].append(t["question_id"])
      ts["labels"].append(t["ml_answer_class"])
      ts["Qs"].append(t["ml_question"])
      ts["As"].append(t["ml_multiple_choice_classes"])
      ts["Is"].append(Is)
      ts["Ts"].append(Ts)
      ts["Gs"].append(Gs)
      ts["ATs"].append(ATs)
      ts["PTs"].append(PTs)
    else: pass # bad id list
  
  for key, value in tr.iteritems(): tr[key] = np.array(value)
  for key, value in ts.iteritems(): ts[key] = np.array(value)
  
  tr["vocabulary"] = vocabulary
  ts["vocabulary"] = vocabulary
  
  tr["classes"] = classes
  ts["classes"] = classes

  pickle.dump(tr, open(os.path.join(outdir, "lr_embedding_tr.p"), "wb"))
  pickle.dump(ts, open(os.path.join(outdir, "lr_embedding_ts.p"), "wb"))
  
  # bow baseline
  modalities = ["qids", "labels", "Is", "Qs", "BoWs", "As"]
  tr = {t:[] for t in modalities}
  ts = {t:[] for t in modalities}
  
  for t in qa:
    album_feat_row_ids = [np.where(aids == k)[0][0] for k in t["album_ids"]]
    if len(album_feat_row_ids) > 1: this_feat = album_feat[album_feat_row_ids,].mean(0)
    else: this_feat = album_feat[album_feat_row_ids[0],]
    
    Ts = [aid2album[k]["ml_album_when"] for k in t["album_ids"]]
    Gs = []
    ATs = []
    PTs = []
    for k in t["album_ids"]: 
      Gs.extend(aid2album[k]["ml_album_where"])
      ATs.extend(aid2album[k]["ml_album_title"])
      PTs = sum(aid2album[k]["ml_photo_titles"], [])
    
    bow = np.zeros(len(vocabulary))
    bow[t["ml_question"]] += 1
    bow[Ts] += 1
    bow[Gs] += 1
    bow[ATs] += 1
    bow[PTs] += 1
    
    divdnorm = np.sum(bow*bow)
    assert  divdnorm != 0 # bow cannot be all zeros
    bow = bow/np.sqrt(divdnorm)

    if t["question_id"] in tr_qids:
      tr["qids"].append(t["question_id"])
      tr["labels"].append(t["ml_answer_class"])
      tr["Qs"].append(t["ml_question"])
      tr["As"].append(t["ml_multiple_choice_classes"])
      tr["Is"].append(this_feat)
      tr["BoWs"].append(bow)
    elif t["question_id"] in ts_qids:
      ts["qids"].append(t["question_id"])
      ts["labels"].append(t["ml_answer_class"])
      ts["Qs"].append(t["ml_question"])
      ts["As"].append(t["ml_multiple_choice_classes"])
      ts["Is"].append(this_feat)
      ts["BoWs"].append(bow)
    else: pass # bad id list

  
  for key, value in tr.iteritems(): tr[key] = np.array(value)
  for key, value in ts.iteritems(): ts[key] = np.array(value)
  
  tr["vocabulary"] = vocabulary
  ts["vocabulary"] = vocabulary
  
  tr["classes"] = classes
  ts["classes"] = classes

  pickle.dump(tr, open(os.path.join(outdir, "bow_tr.p"), "wb"))
  pickle.dump(ts, open(os.path.join(outdir, "bow_ts.p"), "wb"))
  
  # lstm evidence vqa baseline
#   modalities = ["qids", "labels", "Is", "Qs", "As", "CNNFeats"]
#   tr = {t:[] for t in modalities}
#   ts = {t:[] for t in modalities}
#   
#   tr_pids = [t["photo_ids"] for t in album if t["flickr_user_id"] not in test_user_dict]
#   tr_pids = np.unique(sum(tr_pids,[]))
#   pid2featrowid = {pids[i]:i for i in range(len(pids))}
#   
#   
#   tr["CNNFeats"] = photo_feat
#   ts["CNNFeats"] = photo_feat
#   
# 
#   for t in qa:
#         
#     # evidence_photo[pid2featrowid[k] for k in t["evidence_photo_ids"]]
#     if t["flickr_user_id"] not in test_user_dict:
#       tr["qids"].append(t["question_id"])
#       tr["labels"].append(t["ml_answer_class"])
#       tr["Qs"].append(t["ml_question"])
#       tr["As"].append(t["ml_multiple_choice_classes"])
#       tr["Is"].append(this_feat)
#       tr["BoWs"].append(bow)
#     else:
#       ts["qids"].append(t["question_id"])
#       ts["labels"].append(t["ml_answer_class"])
#       ts["Qs"].append(t["ml_question"])
#       ts["As"].append(t["ml_multiple_choice_classes"])
#       ts["Is"].append(this_feat)
#       ts["BoWs"].append(bow)
# 
#   
#   for key, value in tr.iteritems(): tr[key] = np.array(value)
#   for key, value in ts.iteritems(): ts[key] = np.array(value)
#   
#   tr["vocabulary"] = vocabulary
#   ts["vocabulary"] = vocabulary
#   
#   tr["classes"] = classes
#   ts["classes"] = classes
# 
#   pickle.dump(tr, open(os.path.join(outdir, "bow_tr.p"), "wb"))
#   pickle.dump(ts, open(os.path.join(outdir, "bow_ts.p"), "wb"))


  pid2ourfeat = {}
  for t in album:
    this_feat = {}
    for i in range(len(t["photo_ids"])):
      this_feat["PT"] = t["ml_photo_titles"][i]
      this_feat["G"] = t["ml_album_where"]
      this_feat["T"] = t["ml_album_when"]
      this_feat["AT"] = t["ml_album_title"]
      pid2ourfeat[t["photo_ids"][i]] = this_feat 
       
      

  # our model
  modalities = ["qids", "labels", "Is", "Qs", "As", "Ts", "Gs", "PTs", "ATs" ]
  tr = {t:[] for t in modalities}
  ts = {t:[] for t in modalities}
  
  for t in qa:
    evidence_photo_id = t["evidence_photo_ids"][0]
    this_feat = pid2ourfeat[evidence_photo_id]
    Ts = [this_feat["T"]]
    Gs = this_feat["G"]
    ATs = this_feat["AT"]
    PTs = this_feat["PT"]
    

    if t["question_id"] in tr_qids:
      tr["qids"].append(t["question_id"])
      tr["labels"].append(t["ml_answer_class"])
      tr["Qs"].append(t["ml_question"])
      tr["As"].append(t["ml_multiple_choice_classes"])
      #tr["Is"].append(this_feat)
      tr["Ts"].append(Ts)
      tr["Gs"].append(Gs)
      tr["ATs"].append(ATs)
      tr["PTs"].append(PTs)
    elif t["question_id"] in ts_qids:
      ts["qids"].append(t["question_id"])
      ts["labels"].append(t["ml_answer_class"])
      ts["Qs"].append(t["ml_question"])
      ts["As"].append(t["ml_multiple_choice_classes"])
      #ts["Is"].append(this_feat)
      ts["Ts"].append(Ts)
      ts["Gs"].append(Gs)
      ts["ATs"].append(ATs)
      ts["PTs"].append(PTs)
    else: pass # bad id list
  
  for key, value in tr.iteritems(): tr[key] = np.array(value)
  for key, value in ts.iteritems(): ts[key] = np.array(value)
  
  tr["vocabulary"] = vocabulary
  ts["vocabulary"] = vocabulary
  
  tr["classes"] = classes
  ts["classes"] = classes


  pickle.dump(tr, open(os.path.join(outdir, "ours_tr.p"), "wb"))
  pickle.dump(ts, open(os.path.join(outdir, "ours_ts.p"), "wb"))


if __name__ == '__main__':
  build_dataset()