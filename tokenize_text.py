import pandas as pd
import ast
from opt import opt
import numpy as np
#from keras.preprocessing.sequence import pad_sequences
import torch

hash_token = 19
end_token = 20

def pad_sequences(sequences, batch_first=True, padding_value=0.0, max_len=0):
    tmp = torch.Tensor(sequences[0])
    max_size = tmp.size()
    trailing_dims = max_size[1:]
    
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = tmp.data.new(*out_dims).fill_(padding_value)
    for i, list in enumerate(sequences):
        tensor = torch.Tensor(list)
        length = tensor.size(0)
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor.long().numpy()

def set_global_label(bio=False):
    global hash_token
    global end_token 
    if bio:
        hash_token = 3
        end_token = 4
    else:
        hash_token = 2
        end_token = 3

def reg_encoding(cleaned, labels):
    label_l = []
    for oindex, x in enumerate(cleaned):
        tlist = []
        for index, j in enumerate(x):
            for s in j:
                if s[0]=='#':
                    tlist.append(hash_token)
                else:
                    tlist.append(labels[oindex][index])
        label_l.append(tlist)
    return label_l

def bio_encoding(cleaned, labels):
    offset = 1
    
    label_l = []
    for oindex, x in enumerate(cleaned):
        tlist = []
        prev=labels[oindex][0]
        for index, j in enumerate(x):
            #if index==30:
            #ipdb.set_trace()
            for s in j:
                if s[0]=='#':
                    tlist.append(hash_token)
                else:
                    if (index==0 and labels[oindex][index]!=0):
                        tlist.append(labels[oindex][index]+offset)
                        prev = labels[oindex][index]
                    if (prev!=labels[oindex][index] and labels[oindex][index]!= 0):
                        tlist.append(labels[oindex][index]+offset)
                        prev = labels[oindex][index]
                    else:
                        tlist.append(labels[oindex][index])
                        prev = labels[oindex][index]
        label_l.append(tlist)
    return label_l

def concatenate_list_data(cleaned):
    result= []
    for element in cleaned:
        result += element
    return result

def make_set(data_dir, tokenizer, class_type, bio = False):
    dataset = pd.read_csv(data_dir, sep='\t', header=None, converters={1:ast.literal_eval, 2:ast.literal_eval})
    # Shuffle samples
    #dataset = dataset.sample(frac=1)
    if class_type != 'all_class':
        set_global_label(bio)
    terms = list(dataset[1])
    labels = list(dataset[2])
    
    cleaned = [[tokenizer.tokenize(words) for words in sent] for sent in terms]
    tokenized_texts = [concatenate_list_data(sent) for sent in cleaned]

    if bio:
        label_l = bio_encoding(cleaned, labels)
    else:
        label_l = reg_encoding(cleaned, labels)

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          padding_value=0.0, max_len=opt.maxLen)
    #input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
    #                      maxlen=opt.maxLen, dtype="long", truncating="post", padding="post")
    
    tags = pad_sequences(label_l, padding_value=end_token, max_len=opt.maxLen)
    #tags = pad_sequences(label_l,
    #                 maxlen=opt.maxLen, value=end_token, padding="post",
    #                 dtype="long", truncating="post")
    attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
    
    
    return label_l, tokenized_texts, input_ids, tags, attention_masks


