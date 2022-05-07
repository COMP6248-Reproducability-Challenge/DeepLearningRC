'''Preprocessing data for the different GLUE dataset'''

import random
import sys
import os
import csv
import numpy as np 




def _read_tsv(input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


task_data_path = {
    "cola": {"train":"data/CoLA/train.tsv","test":"data/CoLA/test.tsv","dev":"data/CoLA/dev.tsv"},
    "mnli-mm": {"train":"data/MNLI/train.tsv","test":"data/MNLI/test_mismatched.tsv","dev":"data/MNLI/dev_mismatched.tsv"},
    "mnli": {"train":"data/MNLI/train.tsv","test":"data/MNLI/test_matched.tsv","dev":"data/MNLI/dev_matched.tsv"},
    "mrpc": {"train":"data/MRPC/train.tsv","test":"data/MRPC/test.tsv","dev":"data/MRPC/dev.tsv"},
    "sst-2":  {"train":"data/SST-2/train.tsv","test":"data/SST-2/test.tsv","dev":"data/SST-2/dev.tsv"},
    "sts-b":  {"train":"data/STS-B/train.tsv","test":"data/STS-B/test.tsv","dev":"data/STS-B/dev.tsv"},
    "qqp":  {"train":"data/QQP/train.tsv","test":"data/QQP/test.tsv","dev":"data/QQP/dev.tsv"},
    "qnli":  {"train":"data/QNLI/train.tsv","test":"data/QNLI/test.tsv","dev":"data/QNLI/dev.tsv"},
    "rte":  {"train":"data/RTE/train.tsv","test":"data/RTE/test.tsv","dev":"data/RTE/dev.tsv"},
    "wnli":  {"train":"data/WNLI/train.tsv","test":"data/WNLI/test.tsv","dev":"data/WNLI/dev.tsv"},
    "diagnostic": {"test":"data/diagnostic/diagnostic.tsv" }
}


def preprocess_data(task_name, data_type):
    '''Preprocess train or dev data and returns the sentences and labels'''
    data = _read_tsv(task_data_path[task_name][data_type])
    data_dict = {} 
   
    if data_type == 'test':    
    
        if task_name in ['mnli','mnli-mm']:
            id = [x[0]for x in data[1:]]
            sentences_a = [x[8] for x in data[1:]]
            sentences_b = [x[9] for x in data[1:]]
        
        if task_name == 'mrpc':
            id = [x[0]for x in data[1:]]
            sentences_a = [x[3] for x in data[1:]]
            sentences_b = [x[4] for x in data[1:]]
        
        if task_name == 'sts-b':
            id = [x[0]for x in data[1:]]
            sentences_a = [x[7] for x in data[1:]]
            sentences_b = [x[8] for x in data[1:]]

        else: 
            id = [x[0] for x in data[1:]]
            sentences_a = [x[1] for x in data[1:]]
        
        if len(data[0]) == 2:
          data_dict['sentences_a'] = sentences_a
          data_dict['id'] = id
        else:
          sentences_b = [x[2] for x in data[1:]]
          data_dict['sentences_a'] = sentences_a
          data_dict['sentences_b'] = sentences_b
          data_dict['id'] = id



    else:
    
        if task_name == "cola":
            sentences_a = [x[3] for x in data]
            sentences_b = None
            labels = np.array([int(x[1]) for x in data])
        if task_name in ["mnli","mnli-mm"]:
            sentences_a = [x[8] for x in data[1:]]
            sentences_b = [x[9] for x in data[1:]]
            temp_labels = [x[-1] for x in data[1:]]
            data_classes = {'contradiction':0,'entailment':1,'neutral':2}
            labels = np.array([data_classes[k] for k in temp_labels])
        if task_name == "mrpc":
            sentences_a = [x[3] for x in data[1:]]
            sentences_b = [x[4] for x in data[1:]]
            labels = np.array([int(x[0]) for x in data[1:]])
        if task_name == "sst-2":
            sentences_a = [x[0] for x in data[1:]]
            sentences_b = None
            labels = np.array([int(x[1]) for x in data[1:]])
        if task_name == "sts-b":
            sentences_a = [x[7] for x in data[1:]]
            sentences_b = [x[8] for x in data[1:]]
            labels = np.array([x[-1] for x in data[1:]], dtype = float)
        if task_name == "qqp":
            sentences_a = [x[3] for x in data[1:]]
            sentences_b = [x[4] for x in data[1:]]
            labels = np.array([int(x[5]) for x in data[1:]])
        if task_name == "qnli":
            sentences_a = [x[1] for x in data[1:]]
            sentences_b = [x[2] for x in data[1:]]
            temp_labels = [x[-1] for x in data[1:]]
            data_classes = {'not_entailment':0,'entailment':1}
            labels = np.array([data_classes[k] for k in temp_labels])
        if task_name == "rte":
            sentences_a = [x[1] for x in data[1:]]
            sentences_b = [x[2] for x in data[1:]]
            temp_labels = [x[-1] for x in data[1:]]
            data_classes = {'not_entailment':0,'entailment':1}
            labels = np.array([data_classes[k] for k in temp_labels])

        if task_name == "wnli":
            sentences_a = [x[1] for x in data[1:]]
            sentences_b = [x[2] for x in data[1:]]
            labels = np.array([int(x[-1]) for x in data[1:]])
        
        if sentences_b == None:
            data_dict['sentences_a'] = sentences_a
            data_dict['labels'] = labels
        else:
            data_dict['sentences_a'] = sentences_a
            data_dict['sentences_b'] = sentences_b
            data_dict['labels'] = labels

    return data_dict


def preprocess_data_with_rs(task_name, data_type,seed_val = 42):
    '''Preprocess dev and train data to sampleand select the same number of samples as the test dataset'''
    random.seed(seed_val)
    data = _read_tsv(task_data_path[task_name]['dev'])
    data_2 = _read_tsv(task_data_path[task_name]['train'])
    data_dict = {} 

   
    if task_name == "cola":
      if len(data) < 1064:
        data_s = random.sample(data_2, 1064-len(data) )
        data.extend(data_s)
      else:
        data = random.sample(data, 1064)
      sentences_a = [x[3] for x in data]
      sentences_b = None
      labels = np.array([int(x[1]) for x in data])
    if task_name == "mnli-mm":
      if len(data[1:]) < 9848:
        data_s = random.sample(data_2[1:], 9848-len(data[1:]) )
        data.extend(data_s)
        data = data[1:]
      else:
        data = random.sample(data[1:], 9848)
      sentences_a = [x[8] for x in data]
      sentences_b = [x[9] for x in data]
      temp_labels = [x[-1] for x in data]
      data_classes = {'contradiction':0,'entailment':1,'neutral':2}
      labels = np.array([data_classes[k] for k in temp_labels])
   
    if task_name == "mnli":
      if len(data[1:]) < 9797:
        data_s = random.sample(data_2[1:], 9797-len(data[1:]) )
        data.extend(data_s)
        data = data[1:]
      else:
        data = random.sample(data[1:], 9797)

      sentences_a = [x[8] for x in data]
      sentences_b = [x[9] for x in data]
      temp_labels = [x[-1] for x in data]
      data_classes = {'contradiction':0,'entailment':1,'neutral':2}
      labels = np.array([data_classes[k] for k in temp_labels])
    if task_name == "mrpc":
      if len(data[1:]) < 1726:
        data_s = random.sample(data_2[1:], 1726-len(data[1:]) )
        data.extend(data_s)
        data = data[1:]
      else:
        data = random.sample(data[1:], 1726)
  
      sentences_a = [x[3] for x in data]
      sentences_b = [x[4] for x in data]
      labels = np.array([int(x[0]) for x in data])
    if task_name == "sst-2":
      if len(data[1:]) < 1822:
        data_s = random.sample(data_2[1:], 1822-len(data[1:]) )
        data.extend(data_s)
        data = data[1:]
      else:
        data = random.sample(data[1:], 1822)
        
      sentences_a = [x[0] for x in data[1:]]
      sentences_b = None
      labels = np.array([int(float(x[1])) for x in data[1:]])

    if task_name == "sts-b":
      if len(data[1:]) < 1380:
        data_s = random.sample(data_2[1:], 1380-len(data[1:]) )
        data.extend(data_s)
        data = data[1:]
      else:
        data = random.sample(data[1:], 1380)
  
      sentences_a = [x[7] for x in data]
      sentences_b = [x[8] for x in data]
      labels = np.array([x[-1] for x in data], dtype = float)

    if task_name == "qqp":
      if len(data[1:]) < 390966:
        data_s = random.sample(data_2[1:], 390966-len(data[1:]) )
        data.extend(data_s)
        data = data[1:]
      else:
        data = random.sample(data[1:], 390966)
      
      sentences_a = [x[3] for x in data]
      sentences_b = [x[4] for x in data]
      labels = np.array([int(x[5]) for x in data])
    if task_name == "qnli":
      if len(data[1:]) < 5464:
        data_s = random.sample(data_2[1:], 5464-len(data[1:]) )
        data.extend(data_s)
        data = data[1:]
      else:
        data = random.sample(data[1:], 5464)

      sentences_a = [x[1] for x in data]
      sentences_b = [x[2] for x in data]
      temp_labels = [x[-1] for x in data]
      data_classes = {'not_entailment':0,'entailment':1}
      labels = np.array([data_classes[k] for k in temp_labels])
      
    if task_name == "rte":
      data.extend(data[1:])
      data.extend(data_2[1:])
      data.extend(data_2[1:])
      data = random.sample(data[1:], 3001)
        
      sentences_a = [x[1] for x in data]
      sentences_b = [x[2] for x in data]
      temp_labels = [x[-1] for x in data]
      data_classes = {'not_entailment':0,'entailment':1}
      labels = np.array([data_classes[k] for k in temp_labels])

    if task_name == "wnli":
      if len(data[1:]) < 147:
        data_s = random.sample(data_2[1:], 147-len(data[1:]) )
        data.extend(data_s)
        data = data[1:]
      else:
        data = random.sample(data[1:], 147)
      sentences_a = [x[1] for x in data]
      sentences_b = [x[2] for x in data]
      labels = np.array([int(x[-1]) for x in data])
  
    if sentences_b == None:
        data_dict['sentences_a'] = sentences_a
        data_dict['labels'] = labels
    else:
        data_dict['sentences_a'] = sentences_a
        data_dict['sentences_b'] = sentences_b
        data_dict['labels'] = labels

    return data_dict

