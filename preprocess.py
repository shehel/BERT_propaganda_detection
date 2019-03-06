import logging

import pandas as pd

logging.basicConfig(level=logging.INFO)
from collections import defaultdict
import pathlib
import spacy
from pathlib import Path
from spacy.tokens import Doc, Token
nlp = spacy.load('en')


def read_data(directory, isLabels = True):
    ids = []
    texts = []
    labels = []
    for f in directory.glob('*.txt'):
        id = f.name.replace('article', '').replace('.txt','')
        ids.append(id)
        texts.append(f.read_text())
        if isLabels:
            labels.append(parse_label(f.as_posix().replace('.txt', '.task3.labels')))
    docs = list(nlp.pipe(texts))

    return ids, docs, labels

def parse_label(label_path):
    # idx, type, start, end
    labels = []
    f= Path(label_path)
    if not f.exists():
        return labels
    for line in open(label_path):
        parts = line.strip().split('\t')
        labels.append((int(parts[2]), int(parts[3]), parts[1]))
    return sorted(labels)

# Includes additional labels for BIO Encoding
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
]

PT2ID = {y: x for (x, y) in enumerate(PROPAGANDA_TYPES)}

def safe_list_get (l, idx, default=0):
  try:
    return l[idx]
  except IndexError:
    return [0,0, 0]


def bert_list(doc: Doc, doc_labels: list, ids):
    token_idx = 0
    labels_idx = 0
    tokensh = []
    labelsh = []
    # Stores the current sentence and appends to labelsh when current token is
    # \n or \n\n 
    tlabel = []
    tspacyt = []
    ttoken=[]
    bertids = []
    # Variable to store backtrack
    flagger = 0
    spacytokens = []
    current_token: Token = doc[0]
    while token_idx < len(doc):
        current_token: Token = doc[token_idx]
        current_label = safe_list_get(doc_labels, labels_idx)
        if (str(current_token) == '\n' or str(current_token) == '\n\n'):
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
            if flagger == 0:
                ttoken.append(str(current_token))
                tspacyt.append(current_token)
                tlabel.append(0)
            flagger = flagger - 1
            if flagger < 0:
                flagger = 0
            token_idx += 1
            continue
        #ipdb.set_trace()
        start_token_idx = token_idx
        flagger = 0
        while current_token.idx < current_label[1]:
            ttoken.append(str(current_token))
            tspacyt.append(current_token)
            #ipdb.set_trace()
            #print("Marking ", current_token.lower_,'as', PT2ID[current_label[2]])
            tlabel.append(PT2ID[current_label[2]])
            #res[token_idx, PT2ID[current_label[2]]] = label
            token_idx += 1
            if token_idx >= len(doc):
                break
            current_token = doc[token_idx]
            flagger = flagger+1

        # advance label
        labels_idx += 1
            #current_label = safe_list_get(doc_labels, labels_idx)

        # revert token_idx because the labels might be intersecting
        token_idx = start_token_idx
    return bertids, tokensh, labelsh, spacytokens 


if __name__ == '__main__':
    directory = pathlib.Path('./datasets-v5/tasks-2-3/dev/')
    ids, texts, labels = read_data(directory)
    logging.info("Data read")
    
    bertid, bertt, bertl, spacy = zip(*[bert_list(d, l, idx) for d, l, idx in zip(texts, labels, ids)])
    flat_list = [item for sublist in bertt for item in sublist]
    flat_list_l = [item for sublist in bertl for item in sublist]
    flat_list_i = [item for sublist in bertid for item in sublist]
    df = {"ID":flat_list_i, "Tokens":flat_list, "Labels": flat_list_l}
    df = pd.DataFrame(df)

    ds = "dataset_dev.csv"
    df.to_csv(ds, index=False, header=None)
    
    logging.info("Dataset written to %s" % (ds))
