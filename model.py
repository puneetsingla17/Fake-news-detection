from peft.peft_model import PeftConfig
from transformers import BertForSequenceClassification,AdamW,BertConfig,get_linear_schedule_with_warmup,AutoModelForSequenceClassification,AutoTokenizer
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from peft import PeftModel, PeftConfig
import torch
import torch.nn as nn
import json
import random
import numpy as np
import time
import datetime
from utils import get_dataloaders,getdata,dataprep_bert,get_maxlen

def gettokenizer(modelname,lower):
  if lower==0:
    lower=False
  else:
    True
  return AutoTokenizer.from_pretrained(modelname, do_lower_case=lower)


def getmodel(modelname,numlabels,out_attn,out_hiddn,ignore_mis):
  model = AutoModelForSequenceClassification.from_pretrained(
    modelname, # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = numlabels, # The number of output labels--2 for binary classification.
                    
    output_attentions = out_attn, # Whether the model returns attentions weights.
    output_hidden_states = out_hiddn, # Whether the model returns all hidden-states.,
    ignore_mismatched_sizes=ignore_mis
    )

  return model

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):

    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


class trainer:
  
  def __init__(self):
    with open("config.json",'r') as json_file:
          json_data = json.loads(json_file.read())

    modelname=json_data['model_name']
    lower=json_data['lower']
    print(type(lower))
    numlabels=json_data["numlabels"]
    out_attn=json_data['out_attn']
    if out_attn==0:
      out_attn=False
    else:
      out_attn=True

    out_hiddn=bool(json_data['out_hiddn'])
    if out_hiddn==0:
      out_hiddn=False
    else:
      out_hiddn=True
      
    ignore_mis=json_data['ignore_mis']
    if ignore_mis==0:
      ignore_mis=False
    else:
      ignore_mis=True
    
    if json_data['peft']==0:
      self.peft=False
    else:
      self.peft=True
    
    self.path=json_data['path']
    self.batchsize=json_data['batchsize']
    self.tokenizer=gettokenizer(modelname,lower)
    self.model=getmodel(modelname,numlabels,out_attn,out_hiddn,ignore_mis)
    self.epochs=json_data['nepochs']
    self.lr=json_data['learning_rate']
    self.epsilon=json_data["epsilon"]
    self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.model = self.model.to(self.device)
    
    if self.peft:
      self.model=self.convert_peft()
    self.optimizer = AdamW(self.model.parameters(),
                  lr = self.lr, 
                  eps = self.epsilon
                )
  

  def traind(self):
    df=getdata()
    max_len=get_maxlen(df,self.tokenizer)
    input_ids,attention_masks,labels=dataprep_bert(df,self.tokenizer,max_len)
    traindloader,val_dloader=get_dataloaders(input_ids,attention_masks,labels,self.batchsize)
    device=self.device
    print(device)
    total_steps = len(traindloader) * self.epochs
    scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, self.epochs):

        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        total_train_loss = 0
        self.model.train()
        trainlosslist=[]
        for step, batch in enumerate(traindloader):
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the device using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            self.optimizer.zero_grad()
            output = self.model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
            loss = output.loss
            trainlosslist.append(loss.item())
            total_train_loss += loss.item()
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            self.optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(traindloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        print("")
        print("Running Validation...")
        t0 = time.time()
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.model.eval()
        # Tracking variables
        total_eval_accuracy = 0
        best_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        vallosslist=[]
        # Evaluate data for one epoch
        for batch in val_dloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                output= self.model(b_input_ids,
                                      token_type_ids=None,
                                      attention_mask=b_input_mask,
                                      labels=b_labels)
            loss = output.loss
            vallosslist.append(loss.item())
            total_eval_loss += loss.item()
            # Move logits and labels to CPU if we are using GPU
            logits = output.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)
        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(val_dloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(val_dloader)
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        if avg_val_accuracy > best_eval_accuracy:
            if self.peft:
              self.model.save_pretrained(self.path)
            else:
              torch.save(self.model.state_dict(),"bertmodel.pth")
        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
    print("")
    print("Training complete!")
    self.tokenizer.save_pretrained(self.path)
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    return training_stats,trainlosslist,vallosslist


  def convert_peft(self):
    config=LoraConfig(r=32, lora_alpha=32, lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_CLS)
    self.model=get_peft_model(self.model, config)

    print(self.model.print_trainable_parameters())
    return self.model

  def load_peft(self):
    config=PeftConfig.from_pretrained(self.path)
    model=AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model,self.path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    return model,tokenizer

  def inference(self,text):
    model,tokenizer=self.load_peft()
    model.eval()
    print("Peft Model Loaded")

    enc1=tokenizer.batch_encode_plus(text,return_tensors="pt",max_length=256,truncation=True,padding="max_length")
    return model.forward(**enc1)
  
            
      


