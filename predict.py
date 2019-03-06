import logging
from tokenize_text import *

import pickle 
import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import (BasicTokenizer, BertConfig,
                                     BertForTokenClassification, BertTokenizer)
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
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
            if (str(current_token) == '\n' or str(current_token) == '\n\n'):
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

PROPAGANDA_TYPES = [
    "NULL",
    "Appeal_to_Authority",
    "Appeal_to_fear-prejudice",
    "Bandwagon",
    "Black-and-White_Fallacy",
    "Causal_Oversimplification",
    "Doubt",
    "Exaggeration,Minimisation",
    "Flag-Waving",
    "Loaded_Language",
    "Name_Calling,Labeling",
    "Obfuscation,Intentional_Vagueness,Confusion",
    "Red_Herring",
    "Reductio_ad_hitlerum",
    "Repetition",
    "Slogans",
    "Straw_Men",
    "Thought-terminating_Cliches",
    "Whataboutism",
    "Hash",
    "Padding",
    "Appeal_to_Authority",
    "Appeal_to_fear-prejudice",
    "Bandwagon",
    "Black-and-White_Fallacy",
    "Causal_Oversimplification",
    "Doubt",
    "Exaggeration,Minimisation",
    "Flag-Waving",
    "Loaded_Language",
    "Name_Calling,Labeling",
    "Obfuscation,Intentional_Vagueness,Confusion",
    "Red_Herring",
    "Reductio_ad_hitlerum",
    "Repetition",
    "Slogans",
    "Straw_Men",
    "Thought-terminating_Cliches",
    "Whataboutism"
]
def main():
    MAX_LEN = 210
    bs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count(); 
    logging.info("GPUs Detected: %s" % (n_gpu))

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False);
    # Model Initialize
    model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=opt.nLabels);
    
    model.to(device)
    model.load_state_dict(torch.load("./exp/all_class/test/best_model.pth"))
    #directory = pathlib.Path('./data/final/test/')
    #ids, texts, _ = read_data(directory, isLabels=False)
    
    ds = pickle.load( open( "save.p", "rb" ) )

    texts = ds["Text"]
    ids = ds["ID"]

    

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

    numerics = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                            maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

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

    

    end_token = 20

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

    df.to_csv('predictions_localx.csv', sep='\t', index=False, header=False) 
    

if __name__ == '__main__':
    main()
