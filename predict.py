import logging
from tokenize_text import *
from preprocess import *

import pickle 
import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import (BasicTokenizer, BertConfig,
                                     BertForTokenClassification, BertTokenizer)
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as f1
from sklearn.model_selection import train_test_split
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler,
TensorDataset)
from tqdm import tqdm, trange
#import ipdb

import os 
from opt import opt
import itertools
#import subroutine

logging.basicConfig(level=logging.INFO)
def get_spans(a, labelx, i, id_text, hash_token=3, end_token=4):
    #if i==35:
    #ipdb.set_trace()
    spans = []
    span_len = 0
    prev = 0
    for i, x in enumerate(labelx):
        # End if last index\
        
        if x == end_token:
            continue
        if i >= len(a)-1:
            if x != 0:
                # if prev element isn't equal to current and not O
                if prev != x and prev !=0:
                    span_e= a[i-1].idx + len(a[i-1])
                    span_len = 0
                    spans.append([PROPAGANDA_TYPES[labelx[i-1]], span_f, span_e])
                    prev = x
                    span_f = a[i].idx
                    span_len = span_len+1
                if span_len == 0:
                    span_f = a[i].idx
                    span_len = span_len+1
                    prev=x
                    if (i >= len(labelx)-1):
                        span_e= a[i].idx + len(a[i])
                        span_len = 0
                        spans.append([PROPAGANDA_TYPES[labelx[i]], span_f, span_e])
                        continue
                else:
                    span_e= a[i].idx + len(a[i])
                    span_len = 0
                    spans.append([PROPAGANDA_TYPES[labelx[i]], span_f, span_e])
                    continue
                    
            else:
                prev = x
                if (span_len != 0):
                    span_e= a[i-1].idx+len(a[i-1])
                    span_len = 0
                    spans.append([PROPAGANDA_TYPES[labelx[i-1]], span_f, span_e])
                    continue
        if x == hash_token:
            continue
        if x != 0:
            # Check if prev element was same as current or equal to O
            if prev != x and prev !=0:
                span_e= a[i-1].idx + len(a[i-1])
                span_len = 0
                spans.append([PROPAGANDA_TYPES[labelx[i-1]], span_f, span_e])
                prev = x
                span_f = a[i].idx
                span_len = span_len+1
            if span_len == 0:
                span_f = a[i].idx
                span_len = span_len+1
                prev=x
                if (i >= len(labelx)-1):
                    span_e= a[i].idx + len(a[i])
                    span_len = 0
                    spans.append([PROPAGANDA_TYPES[labelx[i]], span_f, span_e])
                    continue
            else:
                if (i >= len(labelx)-1):
                    span_e= a[i].idx + len(a[i])
                    span_len = 0
                    spans.append([PROPAGANDA_TYPES[labelx[i]], span_f, span_e])
                    continue
                span_len = span_len+1

        else:
            prev = x
            if (span_len != 0):
                span_e= a[i-1].idx+len(a[i-1])
                span_len = 0
                spans.append([PROPAGANDA_TYPES[labelx[i-1]], span_f, span_e])
                continue
            if (i >= len(labelx)-1):
                #span_e= a[i].idx + len(a[i])
                #span_len = 0
                #spans.append([span_f, span_e])
                continue
    if spans:
        return id_text, spans
    else:
        return (0, [])
    
def bert_list_test(doc: Doc, ids):
    token_idx = 0
    tokensh = []
    tspacyt = []
    ttoken=[]
    bertids = []
    spacytokens = []
    current_token: Token = doc[0]
    while token_idx < len(doc):
        current_token: Token = doc[token_idx]
        #if (str(current_token) == 'hope'):
        #    ipdb.set_trace()
        if (str(current_token)[:1]) == '\n':
            if ttoken:
                spacytokens.append(tspacyt)
                tokensh.append(ttoken)
                bertids.append(ids)
            tlabel= []
            tspacyt = []
            ttoken=[]
            token_idx += 1
            continue
        ttoken.append(str(current_token))
        tspacyt.append(current_token)
        token_idx += 1

        

            #current_label = safe_list_get(doc_labels, labels_idx)

        # revert token_idx because the labels might be intersecting
    return bertids, tokensh, spacytokens 

def main():
    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4'
    MAX_LEN = 210
    bs = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count(); 
    logging.info("GPUs Detected: %s" % (n_gpu))

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False);
    # Model Initialize
    model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=4);

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(opt.loadModel))
    directory = pathlib.Path('datasets-v5/tasks-2-3/dev/')
    ids, texts, _ = read_data(directory, isLabels=False)
    
    global PROPAGANDA_TYPES    # Needed to modify global copy of globvar
    global PROPAGANDA_TYPES_B

    global hash_token
    global end_token 

    PROPAGANDA_TYPES = [
        "O",
        opt.binaryLabel,
    ]


    hash_token=2
    end_token=3



    bertid, bertt, spacy = zip(*[bert_list_test(d, idx) for d, idx in zip(texts, ids)])

    flat_list = [item for sublist in bertt for item in sublist]
    flat_list_i = [item for sublist in bertid for item in sublist]
    df = {"ID":flat_list_i, "Tokens":flat_list}
    df = pd.DataFrame(df)

    flat_list_s = [item for sublist in spacy for item in sublist]

    ids = list(df["ID"])
    terms = list(df["Tokens"])

    cleaned = [[tokenizer.tokenize(words) for words in sent] for sent in terms]

    tokenized_texts = [concatenate_list_data(sent) for sent in cleaned]

   #numerics = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
    #                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    numerics = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              max_len=MAX_LEN)

    attention_masks = [[float(i>0) for i in ii] for ii in numerics]

    t_inputs = torch.tensor(numerics)
    t_masks = torch.tensor(attention_masks)

    t_data = TensorDataset(t_inputs, t_masks)
    t_dataloader = DataLoader(t_data, sampler=None, batch_size=bs)


    model.eval()
    predictions_sample = []

    for batch in t_dataloader:
        #ipdb.set_trace()
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask= batch
        logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        predictions_sample.extend([list(p) for p in np.argmax(logits, axis=2)])

    
    pred = []
    for oindex, x in enumerate(cleaned):
        index = 0
        tlist = []
        for iindex, j in enumerate(x):
            #print (j)
            #print(index)
            tlist.append(predictions_sample[oindex][index])
            length = len(j)
            index = index + length
            #print ("Token: ", j, "-----  Assigned: ", predictions_sample[oindex][index])
        pred.append(tlist)

    stopp=pred



    pred = []
    for x in stopp:
        tlist = []
        for j in x:
            if j in [hash_token, end_token]:
                continue
            tlist.append(j)
        pred.append(tlist)
    
    print (PROPAGANDA_TYPES)
    lists = []
    liste = []
    listp = []
    listid = []

    for i, x in enumerate(pred):
        a = flat_list_s[i]
        b = flat_list_i[i]
        id_text, spans = get_spans(a, x, i, b)
        if spans:
            for span in spans:
                listid.append(id_text)
                liste.append(span[2])
                lists.append(span[1])
                listp.append(span[0])

    df = {"ID": listid, "P": listp, "s": lists, "liste": liste}

    df = pd.DataFrame(df)

    df.to_csv('pred.csv', sep='\t', index=False, header=False) 
if __name__ == '__main__':
    main()
