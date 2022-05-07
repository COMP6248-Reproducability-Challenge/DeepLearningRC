'''Metrics to evaluate the GLUE dataset on'''


import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import pearsonr, spearmanr

from sklearn.metrics import matthews_corrcoef
from data.preprocess_data import * 
import csv
from pathlib import Path  
import os 


task_file ={ "cola": "CoLA.tsv",
    "mnli-mm": "MNLI-mm.tsv",
    "mnli": "MNLI-m.tsv",
    "mrpc": "MRPC.tsv",
    "sst-2": "SST-2.tsv",
    "sts-b": "STS-B.tsv",
    "qqp": "QQP.tsv",
    "qnli": "QNLI.tsv",
    "rte": "RTE.tsv",
    "wnli":"WNLI.tsv",
    "diagnostic": "AX.tsv"
}

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "sts-b":
        return {"spearman": spearmanr(preds, labels)[0]}
    elif task_name == "qqp":
        return {'acc':simple_accuracy(preds, labels)}
    elif task_name in ["mnli","mnli-mm"]:
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}

    else:
        raise KeyError(task_name)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def convert_tsv_file(preds,task_name, kernel_size, data_type):
    '''To generate .tsv files of the test prediction for submission to GLUE'''
    data_dict = preprocess_data(task_name, data_type='test')
    new_dict = {}
    new_dict['ID'] = data_dict['id']
    if task_name in ['rte','qnli']:
        data_classes = {0:'not_entailment',1: 'entailment'}
        new_dict['label'] = list([data_classes[k] for k in preds])
    if task_name in ['mnli-mm','mnli','diagnostic']:
        data_classes = {0:'contradiction',1:'entailment',2:'neutral'}
        new_dict['label'] = list([data_classes[k] for k in preds])
    if task_name not in ['rte','qnli','mnli-mm','mnli','diagnostic']:
        new_dict['label'] = preds

    new_df =  pd.DataFrame(new_dict)
    sub_dir = os.path.join('glue_scores',str(kernel_size) )
    file_dir = os.path.join(sub_dir,task_file[task_name] )
    os.makedirs(sub_dir, exist_ok = True)
    new_df.to_csv(file_dir, sep="\t", index = False)

    return print("Downloaded Data to " + str(sub_dir))
        

