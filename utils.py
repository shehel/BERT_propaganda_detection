#!/usr/bin/env python
import itertools
from pathlib import Path
import spacy
import pandas as pd
import pathlib
from pathlib import Path
from spacy.tokens import Doc, Token
nlp = spacy.load('en')

def safe_list_get (l, idx, default=0):
  try:
    return l[idx]
  except IndexError:
    return [0,0, 0]

def bert_list_test(doc: Doc, ids:list) -> list:
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

    return bertids, tokensh, spacytokens 

def bert_list(p2id: dict, doc: Doc, doc_labels: list, ids: list, binary: bool, bio: bool = False) -> list:
    if binary:
        offset = 1
    else:
        offset = 18

    token_idx = 0
    labels_idx = 0
    tokensh = []
    labelsh = []
    tlabel = []
    tspacyt = []
    ttoken=[]
    bertids = []
    flagger = 0
    spacytokens = []
    current_token: Token = doc[0]
    while token_idx < len(doc):
        current_token: Token = doc[token_idx]
        start_token_idx = token_idx
        current_label = safe_list_get(doc_labels, labels_idx)
        # advance token until it is within the label
        if (str(current_token)[:1] == '\n'):
            flagger = 0
            if ttoken:
                spacytokens.append(tspacyt)
                tokensh.append(ttoken)
                labelsh.append(tlabel)
                bertids.append(ids)
            tlabel= []
            tspacyt = []
            ttoken=[]
            token_idx += 1
            continue
        if current_token.idx < current_label[0] or current_label[2]==0:
            # Uncomment to get backtrack
            #if flagger == 0:
            ttoken.append(str(current_token))
            tspacyt.append(current_token)
            tlabel.append(0)
            flagger = flagger - 1
            if flagger < 0:
                flagger = 0
            token_idx += 1
            continue
     
        flagger = 0
        first = True
        while current_token.idx < current_label[1]:
            if (str(current_token)[:1] == '\n'):
                if ttoken:
                    spacytokens.append(tspacyt)
                    tokensh.append(ttoken)
                    labelsh.append(tlabel)
                    bertids.append(ids)
                tlabel= []
                tspacyt = []
                ttoken=[]
                
            else: 
                ttoken.append(str(current_token))
                tspacyt.append(current_token)
                            
                if first:
                    tlabel.append(p2id[current_label[2]])
                else:
                     tlabel.append(p2id[current_label[2]] + offset)

            token_idx += 1
            if token_idx >= len(doc):
                break
            current_token = doc[token_idx]
            flagger = flagger+1
            if bio:
                first = False
            else:
                first = True
        # advance label
        labels_idx += 1

        # revert token_idx because the labels might be intersecting. Uncomment to get backtrack.
        #token_idx = start_token_idx
    return bertids, tokensh, labelsh, spacytokens 

def settings(tech_path: str, label: str = None, bio: bool = False) -> list:
    prop_tech_e = None
    if label:
        prop_tech = [label]
    else: 
        prop_tech = load_technique_names_from_file(tech_path)

    if bio:
        prop_tech_inside = prop_tech
        prop_tech_begin = ["B-"+tech for tech in prop_tech]
        prop_tech_inside = ["I-"+tech for tech in prop_tech_inside]
        prop_tech_e = prop_tech_begin + prop_tech_inside
    # TODO prop_tech or prop_tech_e
    offset = len(prop_tech)
    hash_token = offset + 1
    end_token = offset + 2
    # Insert "outside" element
    prop_tech.insert(0, "O")
    if prop_tech_e:
        prop_tech_e.insert(0, "O")
    p2id = {y: x for (x, y) in enumerate(prop_tech)}
    
    return prop_tech_e, prop_tech, hash_token, end_token, p2id

def load_technique_names_from_file(filename: str) -> list:
    with open(filename, "r") as f:
        return [ line.rstrip() for line in f.readlines() ]
        
def read_data(path: str, isLabels: bool = True, binary: bool = None) -> list:
    directory = pathlib.Path(path)
    ids = []
    texts = []
    labels = []
    for f in directory.glob('*.txt'):
        id = f.name.replace('article', '').replace('.txt','')
        ids.append(id)
        texts.append(f.read_text(encoding='utf-8'))
        if isLabels:
            labels.append(parse_label(f.as_posix().replace('.txt', '.task3.labels'), binary=binary))
    docs = list(nlp.pipe(texts))

    return [ids, docs, labels]

def parse_label(label_path: str, binary: bool = None) -> list:
    # idx, type, start, end
    labels = []
    f= Path(label_path)
    if not f.exists():
        return labels
    for line in open(label_path):
        parts = line.strip().split('\t')
        if binary:
            if binary == 'Propaganda':
                labels.append((int(parts[2]), int(parts[3]), 'Propaganda'))
            else:
                if (parts[1] == binary):
                    labels.append((int(parts[2]), int(parts[3]), parts[1]))
        else:
            labels.append((int(parts[2]), int(parts[3]), parts[1]))

    return sorted(labels)
def corpus2list(p2id: dict, ids: list, texts: list, labels: list, binary: bool = False, bio: bool = False) -> list:
    berti, bertt, bertl, berts = zip(*[bert_list(p2id, d, l, idx, binary, bio) for d, l, idx in zip(texts, labels, ids)])
    flat_list_text = [item for sublist in bertt for item in sublist]
    flat_list_label = [item for sublist in bertl for item in sublist]
    flat_list_id = [item for sublist in berti for item in sublist]
    flat_list_spacy = [item for sublist in berts for item in sublist]
    return flat_list_id, flat_list_text, flat_list_label, flat_list_spacy

def test2list(ids: list, texts: list) -> list:
    berti, bertt, berts = zip(*[bert_list_test(d, idx) for d, idx in zip(texts, ids)])
    flat_list_text = [item for sublist in bertt for item in sublist]
    flat_list_id = [item for sublist in berti for item in sublist]
    flat_list_spacy = [item for sublist in berts for item in sublist]
    return flat_list_id, flat_list_text, flat_list_spacy

def get_char_level(flat_list_i: list, flat_list_s: list, predictions_sample: list, cleaned: list, hash_token: int, end_token:int, prop_tech_e) -> pd.DataFrame:
    counter = 0
    for x in predictions_sample:
        for j in x:
            if j == 1:
                counter = counter + 1
                break
    print (counter)
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

    tpred = pred
    pred = []
    for x in tpred:
        tlist = []
        for j in x:
            if j in [hash_token, end_token]:
                continue
            tlist.append(j)
        pred.append(tlist)
    counter = 0
    for x in predictions_sample:
        for j in x:
            if j == 1:
                counter = counter + 1
                break
    print ("Counter check: ", counter)
    lists = []
    liste = []
    listp = []
    listid = []

    for i, x in enumerate(pred):
        a = flat_list_s[i]
        b = flat_list_i[i]
        id_text, spans = get_spans(a, x, i, b, hash_token, end_token, prop_tech_e)
        if spans:
            for span in spans:
                listid.append(id_text)
                liste.append(span[2])
                lists.append(span[1])
                listp.append(span[0])
    df = {"ID": listid, "P": listp, "s": lists, "liste": liste}
    df = pd.DataFrame(df)
    return df

def get_spans(a: list, labelx: list, i: int, id_text: str, hash_token, end_token, prop_tech_e):
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
                    spans.append([prop_tech_e[labelx[i-1]], span_f, span_e])
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
                        spans.append([prop_tech_e[labelx[i]], span_f, span_e])
                        continue
                else:
                    span_e= a[i].idx + len(a[i])
                    span_len = 0
                    spans.append([prop_tech_e[labelx[i]], span_f, span_e])
                    continue
                    
            else:
                prev = x
                if (span_len != 0):
                    span_e= a[i-1].idx+len(a[i-1])
                    span_len = 0
                    spans.append([prop_tech_e[labelx[i-1]], span_f, span_e])
                    continue
        if x == hash_token:
            continue
        if x != 0:
            # Check if prev element was same as current or equal to O
            if prev != x and prev !=0:
                span_e= a[i-1].idx + len(a[i-1])
                span_len = 0
                spans.append([prop_tech_e[labelx[i-1]], span_f, span_e])
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
                    spans.append([prop_tech_e[labelx[i]], span_f, span_e])
                    continue
            else:
                if (i >= len(labelx)-1):
                    span_e= a[i].idx + len(a[i])
                    span_len = 0
                    spans.append([prop_tech_e[labelx[i]], span_f, span_e])
                    continue
                span_len = span_len+1

        else:
            prev = x
            if (span_len != 0):
                span_e= a[i-1].idx+len(a[i-1])
                span_len = 0
                spans.append([prop_tech_e[labelx[i-1]], span_f, span_e])
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
