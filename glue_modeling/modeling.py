'''Implementing ConvBERT for the GLUE Sequence Classification tasks'''

import numpy as np
import random
from torch import rand
from transformers.models.convbert.modeling_convbert import *
from transformers.models.convbert.tokenization_convbert_fast import *
from transformers.models.convbert.tokenization_convbert import *
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from data.preprocess_data import *
from glue_modeling.metrics import *



device = torch.device("cuda")

def tokenization(task_name,model_size,data_type, random_sampling = False):
    if task_name != "squadv1"or "squadv2":
      if random_sampling and data_type == 'dev':
        data_dict = preprocess_data_with_rs(task_name,data_type='dev')
      else:
        data_dict = preprocess_data(task_name,data_type)

  
      model_class, tokenizer_class, pretrained_weights = (ConvBertForSequenceClassification, ConvBertTokenizerFast, model_size)
      tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        
      if len(data_dict.keys()) == 3:
        encoded_dict = tokenizer(data_dict['sentences_a'],data_dict['sentences_b'], add_special_tokens = True, padding = True)
      else:
        encoded_dict = tokenizer(data_dict['sentences_a'], add_special_tokens = True, padding = True)
      
      input_ids = torch.tensor(encoded_dict['input_ids'])
      attention_masks = torch.tensor(encoded_dict['attention_mask'])
      token_type_ids = torch.tensor(encoded_dict['token_type_ids'])
      
      if data_type == 'test':
        return input_ids, attention_masks, token_type_ids
      
      else:
        if task_name == "sts-b":
            
            labels = torch.tensor(data_dict['labels'], dtype = torch.float32)
        else:
            labels = torch.tensor(data_dict['labels'],dtype = torch.long)
        return input_ids, attention_masks, token_type_ids, labels



def dataloader(task_name,data_type,batch_size,model_size, random_sampling = False):
    
    if data_type == 'train':
      input_ids, attention_masks, token_type_ids, labels = tokenization(task_name,model_size,data_type, random_sampling)
      ten_dataset = TensorDataset(input_ids, attention_masks, token_type_ids, labels )
      train_size = int(0.9 * len(ten_dataset))
      val_size = len(ten_dataset) - train_size

        #    Divide the dataset by randomly selecting samples.
      train_dataset, val_dataset = random_split(ten_dataset, [train_size, val_size])

      train_loader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

      val_loader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
      return train_loader,val_loader

    if data_type == 'dev':
      input_ids, attention_masks, token_type_ids, labels = tokenization(task_name,model_size,data_type, random_sampling)
      ten_dataset = TensorDataset(input_ids, attention_masks, token_type_ids, labels )
      prediction_sampler = SequentialSampler(ten_dataset)
      prediction_loader = DataLoader(ten_dataset, sampler=prediction_sampler, batch_size=batch_size)
      return prediction_loader
    
    if data_type == 'test':
      input_ids, attention_masks, token_type_ids = tokenization(task_name,model_size,data_type, random_sampling)
      ten_dataset = TensorDataset(input_ids, attention_masks, token_type_ids)
      prediction_sampler = SequentialSampler(ten_dataset)
      prediction_loader = DataLoader(ten_dataset, sampler=prediction_sampler, batch_size=batch_size)
      return prediction_loader

def train_pred_ft(task_name,model_size,num_epoch=3, seed_val = 42, batch_size = 32, random_sampling = False,kernel_size = 9):
    '''Fine-tuning ConvBERT models for sequence classification on the GLUE dataset'''


    train_loader, val_loader = dataloader(task_name,'train',batch_size,model_size, random_sampling = False)
    pred_loader = dataloader(task_name,'dev',batch_size,model_size, random_sampling )
  
    
    if task_name == "sts-b":
        num_labels = 1
    if task_name in ["mnli-mm","mnli"]:
        num_labels = 3
    if task_name not in ['sts-b','mnli-mm','mnli']: 
        num_labels = 2

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    model = ConvBertForSequenceClassification.from_pretrained(
    model_size, 
    conv_kernel_size = kernel_size,
    num_labels = num_labels,
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False,# Whether the model returns all hidden-states.
   ignore_mismatched_sizes=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                  lr =  3e-4,
                  eps = 1e-6, # args.adam_epsilon 
                  weight_decay = 0.01,
                  betas= (0.9,0.999),
                )
    total_steps = len(train_loader) * num_epoch
    warmup_steps = float(0.1*total_steps)


    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)

   
    training_stats = []


 
    for epoch_i in range(num_epoch):
        
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_epoch))
        print('Training...')

      
        total_train_loss = 0

       
        model.train()

      
        for step, batch in enumerate(train_loader):

        
            if step % 40 == 0 and not step == 0:
            
                
     
                print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_loader)))

            
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device) 
            b_token_type_ids = batch[2].to(device)
            b_labels = batch[3].to(device)
            
            
           
            model.zero_grad()        

            (loss,logits) = model(b_input_ids, 
                                token_type_ids=b_token_type_ids, 
                                attention_mask=b_input_mask, 
                                labels=b_labels,return_dict=False)
            
           
            total_train_loss += loss.item()
            

           
            loss.backward()
            

           
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

           
            optimizer.step()

     
            scheduler.step()


        avg_train_loss = total_train_loss / len(train_loader)            
        
        

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        
            
        # ========================================
        #               Validation
        # ========================================
       

        print("")
        print("Running Validation...")


  
        model.eval()

     
        total_eval_accuracy = 0
        total_eval_loss = 0
   

        # Evaluate data for one epoch
        for batch in val_loader:
            
        
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: token_type_ids
            #   [3]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_token_type_ids = batch[2].to(device)
            b_labels = batch[3].to(device)

            with torch.no_grad():        

               
                (loss,logits) = model(b_input_ids, 
                                    token_type_ids=b_token_type_ids, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels,return_dict = False)
                
                
                
           
            total_eval_loss += loss

            
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

          
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            
        
          
            avg_val_loss = total_eval_loss / len(val_loader)
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))

        
            avg_val_accuracy = total_eval_accuracy / len(val_loader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                
            }
        )

    print("")
    print("Training complete!")


    print('Predicting labels for {:,} test sentences...'.format(len(pred_loader)))


    model.eval()


    predictions , true_labels = [], []


    for batch in pred_loader:
      
        batch = tuple(t.to(device) for t in batch)
        
      
        b_input_ids, b_input_mask, b_token_type_ids , b_labels= batch
        
     
        with torch.no_grad():
          
            outputs = model(b_input_ids, token_type_ids=b_token_type_ids, 
                            attention_mask=b_input_mask)

        logits = outputs[0]
        
      
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
     
        predictions.append(logits)
        true_labels.append(label_ids)

    print('    DONE.')
 
    flat_true_labels = np.concatenate(true_labels, axis=0)


    if task_name != "sts-b":
        for i in range(len(true_labels)):
        
            pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
        
    
        flat_predictions = np.concatenate(predictions, axis=0)
     
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    
    else:
        for i in range(len(true_labels)):
           label with the highest value and turn this
     
            pred_labels_i = predictions[i].flatten()
      
        flat_predictions = np.concatenate(predictions, axis=0)
        
        flat_predictions = flat_predictions.flatten()

    
    return flat_predictions ,flat_true_labels


def train_pred_glue_scores(task_name,model_size,num_epoch=3,  kernel_size = 9, seed_val = 42, batch_size = 32, random_sampling = False):
    '''Predict labels for the test set to be submitted for evaluation on GLUE'''
    
    if task_name != 'diagnostic':
        train_loader, val_loader = dataloader(task_name,'train',batch_size, model_size, random_sampling= False )
    
    pred_loader = dataloader(task_name,'test',batch_size,model_size, random_sampling= False )
    
    if task_name == "sts-b":
        num_labels = 1
    elif task_name == "mnli-mm" or "mnli":
        num_labels = 3
    else: 
        num_labels = 2
    
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    model = ConvBertForSequenceClassification.from_pretrained(
    model_size, 
    conv_kernel_size = kernel_size,
    num_labels = num_labels,
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False,# Whether the model returns all hidden-states.
   ignore_mismatched_sizes=True).to(device)

    if task_name != 'diagnostic':
        optimizer = torch.optim.AdamW(model.parameters(),
                  lr =  3e-4,
                  eps = 1e-6, # args.adam_epsilon 
                  weight_decay = 0.01,
                  betas= (0.9,0.999),
                )
        total_steps = len(train_loader) * num_epoch
        warmup_steps = float(0.1*total_steps)

  
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)

    if task_name == 'diagnostic':
        model.eval()
  
        predictions , true_labels = [], []


        for batch in pred_loader:
          
            batch = tuple(t.to(device) for t in batch)
            
          
            b_input_ids, b_input_mask, b_token_type_ids = batch
            
           
            with torch.no_grad():
             
                outputs = model(b_input_ids, token_type_ids=b_token_type_ids, 
                                attention_mask=b_input_mask)

            logits = outputs[0]
            
        
            logits = logits.detach().cpu().numpy()
        
            

            predictions.append(logits)

        print('    DONE.')
        
 
        flat_predictions = np.concatenate(predictions, axis=0)
        
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        
        return flat_predictions 

    else:
    
        training_stats = []


      
        for epoch_i in range(num_epoch):
            
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_epoch))
            print('Training...')

         
            total_train_loss = 0

         
            model.train()

            for step, batch in enumerate(train_loader):

              
                if step % 40 == 0 and not step == 0:
                
              
         
                    print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_loader)))

              
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device) 
                b_token_type_ids = batch[2].to(device)
                b_labels = batch[3].to(device)

                
                
                model.zero_grad()        

             
                (loss,logits) = model(b_input_ids, 
                                    token_type_ids=b_token_type_ids, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels,return_dict=False)
            

                total_train_loss += loss.item()
                

              
                loss.backward()
                

      
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

             
                scheduler.step()

           
            avg_train_loss = total_train_loss / len(train_loader)            
            
  

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            
                
            # ========================================
            #               Validation
            # ========================================
          

            print("")
            print("Running Validation...")


          
            model.eval()

    
            total_eval_accuracy = 0
            total_eval_loss = 0
        

          
            for batch in val_loader:
                
             
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: token_type_ids
                #   [3]: labels 
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_token_type_ids = batch[2].to(device)
                b_labels = batch[3].to(device)
                
              
                with torch.no_grad():        

                  
                    (loss,logits) = model(b_input_ids, 
                                        token_type_ids=b_token_type_ids, 
                                        attention_mask=b_input_mask,
                                        labels=b_labels,return_dict = False)
                    
                    
                    
    
                total_eval_loss += loss

              
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

            
          
                total_eval_accuracy += flat_accuracy(logits, label_ids)
                
            
            
                avg_val_loss = total_eval_loss / len(val_loader)
                print("  Validation Loss: {0:.2f}".format(avg_val_loss))

       
                avg_val_accuracy = total_eval_accuracy / len(val_loader)
                print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

         
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    
                }
            )

        print("")
        print("Training complete!")


        print('Predicting labels for {:,} test sentences...'.format(len(pred_loader)))

 
        model.eval()

     
        predictions , true_labels = [], []


        for batch in pred_loader:
           
            batch = tuple(t.to(device) for t in batch)
            
           
            b_input_ids, b_input_mask, b_token_type_ids = batch
            
          
            with torch.no_grad():
              
                outputs = model(b_input_ids, token_type_ids=b_token_type_ids, 
                                attention_mask=b_input_mask)

            logits = outputs[0]
            
 
            logits = logits.detach().cpu().numpy()
        
            
        
            predictions.append(logits)



        print('    DONE.')

        
            
        if task_name != "sts-b":
            
            
           
            
            flat_predictions = np.concatenate(predictions, axis=0)
           
            flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        
        else:
           
            flat_predictions = np.concatenate(predictions, axis=0)
           
            flat_predictions = flat_predictions.flatten()
     
        return flat_predictions

