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

def get_spans(a, labelx, i, id_text):
        #if i==35:
        #ipdb.set_trace()
        spans = []
        span_len = 0
        prev = 0
        for i, x in enumerate(labelx):
            if i >= len(a)-1:
                break
            if x == 19:
                continue
            if x == 20:
                continue
            if x != 0:
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

def bert_list_test(doc, ids):
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
            if (str(current_token)[:1] == '\n'):
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

def set_globals(label):
    global PROPAGANDA_TYPES    # Needed to modify global copy of globvar
    global PROPAGANDA_TYPES_B
    PROPAGANDA_TYPES = [
    "O", label
    ]
    PROPAGANDA_TYPES_B = [
    "O",
    "B-"+label,
    "I-"+label,
    ]
    global hash_token
    hash_token = 2
    global end_token 
    end_token = 3

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count(); 
    logging.info("GPUs Detected: %s" % (n_gpu))

    if (opt.classType != "all_class"):
        set_globals(opt.binaryLabel)

    tokenizer = BertTokenizer.from_pretrained(opt.model, do_lower_case=opt.lowerCase);
    # Model Initialize
    model = BertForTokenClassification.from_pretrained(opt.model, num_labels=opt.nLabels);
    
    model.to(device)
    model.load_state_dict(torch.load(opt.loadModel))
    
    
    directory = pathlib.Path(opt.valDataset)
    ids, texts, _ = read_data(directory, isLabels=False)
    
    bertid, bertt, spacy = zip(*[bert_list_test(d, idx) for d, idx in zip(texts, ids)])

    terms = [item for sublist in bertt for item in sublist]
    ids = [item for sublist in bertid for item in sublist]
    
    spacy_sentence = [item for sublist in spacy for item in sublist]

    cleaned = [[tokenizer.tokenize(words) for words in sent] for sent in terms]

    tokenized_texts = [concatenate_list_data(sent) for sent in cleaned]

    numerics = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                            maxlen=opt.maxLen, dtype="long", truncating="post", padding="post")

    attention_masks = [[float(i>0) for i in ii] for ii in numerics]

    t_inputs = torch.tensor(numerics)
    t_masks = torch.tensor(attention_masks)

    t_data = TensorDataset(t_inputs, t_masks)
    t_dataloader = DataLoader(t_data, sampler=None, batch_size=opt.validBatch)

    model.eval()
    predictions_sample = []

    for batch in tqdm(t_dataloader, desc="Predicting"):
        #ipdb.set_trace()
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask= batch
        logits = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        predictions_sample.extend([list(p) for p in np.argmax(logits, axis=2)])
   

    pred = []
    for x in predictions_sample:
        tlist = []
        for j in x:
            if j in [hash_token, end_token]:
                continue
            tlist.append(j)
        pred.append(tlist)

    lists = []
    liste = []
    listp = []
    listid = []

    for i, x in enumerate(pred):
        a = spacy_sentence[i]
        b = ids[i]
        id_text, spans = get_spans(a, x, i, b)
        if spans:
            for span in spans:
                listid.append(id_text)
                liste.append(span[2])
                lists.append(span[1])
                listp.append(span[0])

    df = {"ID": listid, "P": listp, "s": lists, "liste": liste}

    df = pd.DataFrame(df)

    df.to_csv(opt.outputFile, sep='\t', index=False, header=False) 
    
    logging.info("Predictions written to: %s" % (opt.outputFile))

#    subprocess.call("tools/task3_scorer_onefile.py -r ../datasets-v5/tasks-2-3/dev/ -s ../"+opt.outputFile, shell=True)

if __name__ == '__main__':
    main()
