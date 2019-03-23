import logging

import pandas as pd
import argparse
logging.basicConfig(level=logging.INFO)
from collections import defaultdict
import pathlib
import spacy
from pathlib import Path
from spacy.tokens import Doc, Token
nlp = spacy.load('en')

def read_data(directory, isLabels = True, binary=None):
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

    return ids, docs, labels

def parse_label(label_path, binary=None):
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


# Includes additional labels for BIO Encoding
PROPAGANDA_TYPES = [
    "O",
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

PROPAGANDA_TYPES_B = [
    "O",
    "B-Appeal_to_Authority",
    "B-Appeal_to_fear-prejudice",
    "B-Bandwagon",
    "B-Black-and-White_Fallacy",
    "B-Causal_Oversimplification",
    "B-Doubt",
    "B-Exaggeration,Minimisation",
    "B-Flag-Waving",
    "B-Loaded_Language",
    "B-Name_Calling,Labeling",
    "B-Obfuscation,Intentional_Vagueness,Confusion",
    "B-Red_Herring",
    "B-Reductio_ad_hitlerum",
    "B-Repetition",
    "B-Slogans",
    "B-Straw_Men",
    "B-Thought-terminating_Cliches",
    "B-Whataboutism",
    "I-Appeal_to_Authority",
    "I-Appeal_to_fear-prejudice",
    "I-Bandwagon",
    "I-Black-and-White_Fallacy",
    "I-Causal_Oversimplification",
    "I-Doubt",
    "I-Exaggeration,Minimisation",
    "I-Flag-Waving",
    "I-Loaded_Language",
    "I-Name_Calling,Labeling",
    "I-Obfuscation,Intentional_Vagueness,Confusion",
    "I-Red_Herring",
    "I-Reductio_ad_hitlerum",
    "I-Repetition",
    "I-Slogans",
    "I-Straw_Men",
    "I-Thought-terminating_Cliches",
    "I-Whataboutism"
]
def set_global_vars(label):
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
    global PT2ID
    PT2ID = {y: x for (x, y) in enumerate(PROPAGANDA_TYPES)}

PT2ID = {y: x for (x, y) in enumerate(PROPAGANDA_TYPES)}

def safe_list_get (l, idx, default=0):
  try:
    return l[idx]
  except IndexError:
    return [0,0, 0]


def bert_list(doc: Doc, doc_labels: list, ids, binary, bio=False):
    if binary:
        offset = 18
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
                    tlabel.append(PT2ID[current_label[2]])
                else:
                    tlabel.append(PT2ID[current_label[2]]+offset)

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

def main(args):

    directory = pathlib.Path(args.dataset)
    if (args.binary):
        set_global_vars(args.binary)
    ids, texts, labels = read_data(directory, binary=args.binary)
    logging.info("Data read")
    
    bertid, bertt, bertl, spacy = zip(*[bert_list(d, l, idx, args.binary, args.bio) for d, l, idx in zip(texts, labels, ids)])
    flat_list = [item for sublist in bertt for item in sublist]
    flat_list_l = [item for sublist in bertl for item in sublist]
    flat_list_i = [item for sublist in bertid for item in sublist]
    if args.bio == None:
        df = {"ID":flat_list_i, "Tokens":flat_list, "Labels": flat_list_l}
    else: 
        #encoded = bio_encoding(flat_list_l)
        bio = []
        bio_l = []
        bio_ids = []
        count = 1
        prev = flat_list_i[0]
        for i,x,y in zip(flat_list_i, flat_list, flat_list_l):
            if i != prev:
                count = 1
            for token, label in zip(x, y):
                bio_ids.append(str(i)+'_'+str(count))
                bio.append(token)
                bio_l.append(PROPAGANDA_TYPES_B[int(label)])
            bio_ids.append('')
            bio.append('')
            bio_l.append('')
            count = count + 1
            prev = i
        
        df = {"Token":bio, "Label": bio_l}
        logging.info("Data in BIO Format")

    df = pd.DataFrame(df)

    ds = args.output
    df.to_csv(ds, index=False, header=None, sep='\t')
    
    logging.info("Dataset written to %s" % (ds))

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Preprocessing step to obtain model compatible inputs")
    parser.add_argument('-d', '--dataset-dir', dest='dataset', required=True, help="Directory containing the articles and labels.")
    parser.add_argument('-o', '--output-file', dest='output', required=True, help="Name of the file to store output to")
    parser.add_argument('-s', '--binary', dest='binary', required=False, help="Provide the name of the label if the task is binary classification")
    parser.add_argument('-b', '--bio', dest='bio', required=False, help="Use this flag to get output in coNLL-2002 format", type=bool, nargs="?", const=True)

    main(parser.parse_args())
