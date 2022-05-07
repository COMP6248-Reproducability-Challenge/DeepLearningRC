'''Implementing ConvBERT for the GLUE Sequence Classification tasks'''

import numpy as np
import random
from torch import rand
from transformers.models.convbert.modeling_convbert import *
from transformers.models.convbert.tokenization_convbert_fast import *
from transformers.models.convbert.tokenization_convbert import *
from data.preprocess_data import *
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from metrics import *



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
            labels = torch.tensor(data_dict['labels'])
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

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)

    #start training 
    
    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy,
    training_stats = []


    # For each epoch...
    for epoch_i in range(num_epoch):
        
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_epoch))
        print('Training...')

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_loader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
            
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_loader)))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device) 
            b_token_type_ids = batch[2].to(device)
            b_labels = batch[3].to(device)
            
            
            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            (loss,logits) = model(b_input_ids, 
                                token_type_ids=b_token_type_ids, 
                                attention_mask=b_input_mask, 
                                labels=b_labels,return_dict=False)
            
            
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item()
            

            # Perform a backward pass to calculate the gradients.
            loss.backward()
            

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_loader)            
        
        # Measure how long this epoch took.

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        
            
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")


        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in val_loader:
            
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: token_type_ids
            #   [3]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_token_type_ids = batch[2].to(device)
            b_labels = batch[3].to(device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                (loss,logits) = model(b_input_ids, 
                                    token_type_ids=b_token_type_ids, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels,return_dict = False)
                
                
                
            # Accumulate the validation loss.
            total_eval_loss += loss

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            
        
            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(val_loader)
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))

        # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(val_loader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Record all statistics from this epoch.
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

    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []

    # Predict 
    for batch in pred_loader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_token_type_ids , b_labels= batch
        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=b_token_type_ids, 
                            attention_mask=b_input_mask)

        logits = outputs[0]
        
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    print('    DONE.')
    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)


    if task_name != "sts-b":
        for i in range(len(true_labels)):
            # The predictions for this batch are a 2-column ndarray (one column for "0" 
            # and one column for "1"). Pick the label with the highest value and turn this
            # in to a list of 0s and 1s.
            pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
        
        # Combine the results across all batches. 
        flat_predictions = np.concatenate(predictions, axis=0)
        # For each sample, pick the label (0 or 1) with the higher score.
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    
    else:
        for i in range(len(true_labels)):
            # The predictions for this batch are a 2-column ndarray (one column for "0" 
            # and one column for "1"). Pick the label with the highest value and turn this
            # in to a list of 0s and 1s.
            pred_labels_i = predictions[i].flatten()
        # Combine the results across all batches. 
        flat_predictions = np.concatenate(predictions, axis=0)
        # For each sample, pick the label (0 or 1) with the higher score.
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

    # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)

    if task_name == 'diagnostic':
        model.eval()
        # Tracking variables 
        predictions , true_labels = [], []

        # Predict 
        for batch in pred_loader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_token_type_ids = batch
            
            # Telling the model not to compute or store gradients, saving memory and 
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(b_input_ids, token_type_ids=b_token_type_ids, 
                                attention_mask=b_input_mask)

            logits = outputs[0]
            
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
        
            
            # Store predictions and true labels
            predictions.append(logits)

        print('    DONE.')
        
        # Combine the results across all batches. 
        flat_predictions = np.concatenate(predictions, axis=0)
        # For each sample, pick the label (0 or 1) with the higher score.
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        
        return flat_predictions 

    else:
    #start training 
    
    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy,
        training_stats = []


        # For each epoch...
        for epoch_i in range(num_epoch):
            
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_epoch))
            print('Training...')

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode. Don't be mislead--the call to 
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_loader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                
                    
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_loader)))

                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the 
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device) 
                b_token_type_ids = batch[2].to(device)
                b_labels = batch[3].to(device)

                
                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because 
                # accumulating the gradients is "convenient while training RNNs". 
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                model.zero_grad()        

                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # arge given and what flags are set. For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                (loss,logits) = model(b_input_ids, 
                                    token_type_ids=b_token_type_ids, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels,return_dict=False)
            

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value 
                # from the tensor.
                total_train_loss += loss.item()
                

                # Perform a backward pass to calculate the gradients.
                loss.backward()
                

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_loader)            
            
            # Measure how long this epoch took.

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            
                
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")


            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables 
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in val_loader:
                
                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using 
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: token_type_ids
                #   [3]: labels 
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_token_type_ids = batch[2].to(device)
                b_labels = batch[3].to(device)
                
                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():        

                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which 
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here: 
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    (loss,logits) = model(b_input_ids, 
                                        token_type_ids=b_token_type_ids, 
                                        attention_mask=b_input_mask,
                                        labels=b_labels,return_dict = False)
                    
                    
                    
                # Accumulate the validation loss.
                total_eval_loss += loss

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
          
                total_eval_accuracy += flat_accuracy(logits, label_ids)
                
            
                # Calculate the average loss over all of the batches.
                avg_val_loss = total_eval_loss / len(val_loader)
                print("  Validation Loss: {0:.2f}".format(avg_val_loss))

            # Report the final accuracy for this validation run.
                avg_val_accuracy = total_eval_accuracy / len(val_loader)
                print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Record all statistics from this epoch.
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

        # Put model in evaluation mode
        model.eval()

        # Tracking variables 
        predictions , true_labels = [], []

        # Predict 
        for batch in pred_loader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_token_type_ids = batch
            
            # Telling the model not to compute or store gradients, saving memory and 
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(b_input_ids, token_type_ids=b_token_type_ids, 
                                attention_mask=b_input_mask)

            logits = outputs[0]
            
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
        
            
            # Store predictions and true labels
            predictions.append(logits)



        print('    DONE.')

        
            
        if task_name != "sts-b":
            
            
            # Combine the results across all batches. 
            
            flat_predictions = np.concatenate(predictions, axis=0)
            # For each sample, pick the label (0 or 1) with the higher score.
            flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        
        else:
           
            flat_predictions = np.concatenate(predictions, axis=0)
            #flat_predictions = sts_scaler.inverse_transform(temp_predictions.reshape(-1,1))
            # For each sample, pick the label (0 or 1) with the higher score.
            flat_predictions = flat_predictions.flatten()
            print(flat_predictions)
        return flat_predictions

