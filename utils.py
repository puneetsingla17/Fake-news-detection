from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,random_split
import torch

def getdata():
  dataset = load_dataset("datacommons_factcheck", "fctchk_politifact_wapo")

  d=dataset['train']
  df=d.data.to_pandas()
  df.review_rating=df.review_rating.apply(lambda x:x.lower())
  s=set(df.review_rating.value_counts()[:40].index)

  fclasses=['false','mostly false','pants on fire','half true','four pinocchios','three pinocchios','distorts the facts','misleading','no evidence','spins the facts','full flop','unsupported','exaggerates','one pinocchio','wrong',
            'exagerated','not quite right','disputed','in dispute','spinning the facts','false: distorts facts','under dispute']

  removeclasses=['verdict pending','we explain the numbers','no way to know']

  fs=set(fclasses).union(set(removeclasses))

  ts=s-fs
  ts=ts.union({'one pinocchios','half true'})

  s1=set(fclasses).union(ts)

  df=df[df.review_rating.isin(s1)]

  df['label']=df.apply(lambda x:labelfunc(x,ts),axis=1)
  return df

def labelfunc(l,ts):
  if l['review_rating'] in ts:
    return 1
  return 0


def dataprep_bert(df,tokenizer,max_len):
  input_ids = []
  attention_masks = []


  for sent in df.claim_text:
      # `encode_plus` will:
      #   (1) Tokenize the sentence.
      #   (2) Prepend the `[CLS]` token to the start.
      #   (3) Append the `[SEP]` token to the end.
      #   (4) Map tokens to their IDs.
      #   (5) Pad or truncate the sentence to `max_length`
      #   (6) Create attention masks for [PAD] tokens.
      encoded_dict = tokenizer.encode_plus(
                          sent,                      # Sentence to encode.
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                          max_length = max_len,           # Pad & truncate all sentences.
                          pad_to_max_length = True,
                          return_attention_mask = True,   # Construct attn. masks.
                          return_tensors = 'pt',     # Return pytorch tensors.
                    )

      # Add the encoded sentence to the list.
      input_ids.append(encoded_dict['input_ids'])

      # And its attention mask (simply differentiates padding from non-padding).
      attention_masks.append(encoded_dict['attention_mask'])

  # Convert the lists into tensors.
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  labels = torch.tensor(df.label.values)

  return input_ids,attention_masks,labels


def get_dataloaders(input_ids,attention_masks,labels,batch_size):
  dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a 80-20 train-validation split.

# Calculate the number of samples to include in each set.
  train_size = int(0.80 * len(dataset))
  #val_size = int(0.20 * len(dataset))
  val_size = len(dataset)  - train_size

  print("trainsize ",train_size)
  print("validsize ",val_size)
  # Divide the dataset by randomly selecting samples.
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
  train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

  # For validation the order doesn't matter, so we'll just read them sequentially.
  validation_dataloader = DataLoader(
              val_dataset, # The validation samples.
              sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
              batch_size = batch_size) # Evaluate with this batch size.

  return train_dataloader,validation_dataloader
  

def get_maxlen(df,tokenizer):
  max_len = 0

  # For every sentence...
  for sent in df.claim_text:

      # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
      input_ids = tokenizer.encode(sent, add_special_tokens=True)

      # Update the maximum sentence length.
      max_len = max(max_len, len(input_ids))
  return max_len
  
