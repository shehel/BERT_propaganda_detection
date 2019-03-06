import pandas as pd
import ast
from opt import opt
import numpy as np
from keras.preprocessing.sequence import pad_sequences

hash_token = 19
end_token = 20


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
                #else:
                 #   tlist.append(labels[oindex][index])
        label_l.append(tlist)
    return label_l

def bio_encoding(cleaned, labels):
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
                        tlist.append(labels[oindex][index]+20)
                        prev = labels[oindex][index]
                    if (prev!=labels[oindex][index] and labels[oindex][index]!= 0):
                        tlist.append(labels[oindex][index]+20)
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

def make_set(data_dir, tokenizer):
    dataset = pd.read_csv(data_dir, sep=',', header=None, converters={1:ast.literal_eval, 2:ast.literal_eval})
    # Shuffle samples
    dataset = dataset.sample(frac=1)
    
    terms = list(dataset[1])
    labels = list(dataset[2])
    
    cleaned = [[tokenizer.tokenize(words) for words in sent] for sent in terms]
    tokenized_texts = [concatenate_list_data(sent) for sent in cleaned]

    label_l = reg_encoding(cleaned, labels)
    
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=opt.maxLen, dtype="long", truncating="post", padding="post")
    
    tags = pad_sequences(label_l,
                     maxlen=opt.maxLen, value=end_token, padding="post",
                     dtype="long", truncating="post")
    attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
    
    
    return label_l, tokenized_texts, input_ids, tags, attention_masks


