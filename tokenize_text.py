import pandas as pd
import pickle
import ast
from opt import opt
import numpy as np
import torch
from utils import *


def pad_sequences(sequences: list, batch_first: bool = True, padding_value: int = 0, max_len: int = 0):
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

# def set_global_label(bio: bool = False) -> None:
#     global hash_token
#     global end_token 
#     if bio:
#         hash_token = 3
#         end_token = 4
#     else:
#         hash_token = 2
#         end_token = 3

def reg_encoding(cleaned: list, labels: list, hash_token, end_token) -> list:
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

def bio_encoding(cleaned: list, labels: list) -> list:
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

def concatenate_list_data(cleaned: list) -> list:
    result= []
    for element in cleaned:
        result += element
    return result

def make_set(p2id, data_dir: str, tokenizer, single_class: str, 
             hash_token, end_token, bio: bool = False) -> list: 
    #dataset = pd.read_csv(data_dir, sep='\t', header=None, converters={1:ast.literal_eval, 2:ast.literal_eval})
    data_dict = pickle.load(open(data_dir, "rb"))
    
    dataset = corpus2list(p2id, data_dict["ID"], data_dict["Text"],
                              data_dict["Label"], single_class, bio)
    # Shuffle samples
    #dataset = dataset.sample(frac=1)
    terms = list(dataset[1])
    labels = list(dataset[2])
    
    cleaned = [[tokenizer.tokenize(words) for words in sent] for sent in terms]
    tokenized_texts = [concatenate_list_data(sent) for sent in cleaned]
    if bio:
        label_l = bio_encoding(cleaned, labels)
    else:
        label_l = reg_encoding(cleaned, labels, hash_token, end_token)

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          padding_value=0.0, max_len=opt.maxLen)
    
    tags = pad_sequences(label_l, padding_value=end_token, max_len=opt.maxLen)
    attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
    
    
    return input_ids, tags, attention_masks, label_l


def make_val_set(p2id, data_dir: str, tokenizer, single_class: str, 
             hash_token, end_token, bio: bool = False) -> list: 
    #dataset = pd.read_csv(data_dir, sep='\t', header=None, converters={1:ast.literal_eval, 2:ast.literal_eval})
    data_dict = pickle.load(open(data_dir, "rb"))
    if not bio:
        dataset = corpus2list(p2id, data_dict["ID"], data_dict["Text"],
                              data_dict["Label"], single_class, bio)
    # Shuffle samples
    #dataset = dataset.sample(frac=1)
    ids = (dataset[0])
    terms = (dataset[1])
    labels = (dataset[2])
    spacy = (dataset[3])
    cleaned = [[tokenizer.tokenize(words) for words in sent] for sent in terms]
    tokenized_texts = [concatenate_list_data(sent) for sent in cleaned]

    if bio:
        label_l = bio_encoding(cleaned, labels)
    else:
        label_l = reg_encoding(cleaned, labels, hash_token, end_token)

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          padding_value=0.0, max_len=opt.maxLen)
    
    tags = pad_sequences(label_l, padding_value=end_token, max_len=opt.maxLen)
    attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
    
    
    return input_ids, tags, attention_masks, cleaned, ids, terms, spacy, label_l
