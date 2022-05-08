
import argparse
import collections
import json
import statistics

from data.preprocess_data import *
from glue_modeling.modeling import *
from glue_modeling.metrics import *
import random



model_sizes = {'small':'YituTech/conv-bert-small',
            'medium-small':'YituTech/conv-bert-medium-small',
            'base': 'YituTech/conv-bert-base'}



def run_glue_experiments(model_size,task_name,experiment_type,longerthan4M,kernel_size):
  ms= model_sizes[model_size]
  if experiment_type == 'fine-tuning': #fine-tuning for experiment with predictions on the data with the same size as the test set
      if not longerthan4M:
          num_epoch = 3
      if longerthan4M:
          num_epoch = 5

      pred, labels = train_pred_ft(task_name = task_name , model_size = ms, num_epoch = num_epoch, seed_val = 42, batch_size = 32,random_sampling = True,kernel_size=kernel_size)
      metrics_dict = compute_metrics(task_name,pred,labels)
      print(metrics_dict)
    


  if experiment_type == 'dev-pred': #dev-pred for experiment with predictions on the dev data to get the glue scores on dev sets
    all_tasks =["cola","mnli","mrpc","sst-2","sts-b","qqp", "qnli","rte"]
    glue_score = []
    for task_name in all_tasks: 
      pred, labels = train_pred_ft(task_name = task_name , model_size = ms, num_epoch = 3, seed_val = 42, batch_size = 32,random_sampling = False, kernel_size = kernel_size)
      metrics_dict = compute_metrics(task_name,pred,labels)
      glue_score.append(metrics_dict[list(metrics_dict.keys())[0]])
  
    return print("glue score: ", np.mean(glue_score))
  
  if experiment_type == 'glue-scoring': #glue-scoring to achieve labels in .tsv file for submission to GLUE
  
      pred = train_pred_glue_scores(task_name= task_name, model_size = ms, num_epoch=3, kernel_size = kernel_size,seed_val = 42, batch_size = 32, random_sampling=False)
      convert_tsv_file(pred,task_name,kernel_size,data_type = 'test')
    

  if experiment_type == '9-training-rounds': #9-training-rounds to obtain median results on GLUE dev set 
      temp_metrics = []
      for i in range(9):
        pred, labels = train_pred_ft(task_name = task_name, model_size = ms, num_epoch=3, seed_val =  random.randint(42,420)
      , batch_size = 32, random_sampling=False,kernel_size=kernel_size)
        metrics_dict = compute_metrics(task_name,pred,labels)
        temp_metrics.append(metrics_dict[list(metrics_dict.keys())[0]])
        print(temp_metrics)

      return print(statistics.median(temp_metrics))
  


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model_size", required=True,
                      help="small,medium-small or base")
  parser.add_argument("--task_name", required=False,
                      help="name of the dataset")
  parser.add_argument("--experiment_type", required = True,
                      help="glue-scoring or finetuning or dev-pred or 9-training-rounds or question-answering")
  parser.add_argument("--kernel_size", type = int, default = 9,
                      help="specify kernel size if experiment_type == glue-scoring or dev-pred")
  parser.add_argument("--longerthan4M", type = bool, default = False,
                      help="specify if finetuning for longer than 4M updates")
  


  args = parser.parse_args()
 
  run_glue_experiments(args.model_size, args.task_name,args.experiment_type, args.longerthan4M, args.kernel_size)


if __name__ == "__main__":
  main()
