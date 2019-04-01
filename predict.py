import logging
from tokenize_text import *
from preprocess import *
import tools.task3_scorer_onefile
from utils import *
import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import (BasicTokenizer, BertConfig,
                                     BertForTokenClassification, BertTokenizer)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler,
TensorDataset)
from tqdm import tqdm, trange
#import ipdb

import os 
from opt import opt
import itertools
#import subroutine

logging.basicConfig(level=logging.INFO)
    

def main():
    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4'
    
    prop_tech_e, prop_tech, hash_token, end_token, p2id = settings(opt.techniques, opt.binaryLabel, opt.bio)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count(); 
    logging.info("GPUs Detected: %s" % (n_gpu))

    tokenizer = BertTokenizer.from_pretrained(opt.model, do_lower_case=opt.lowerCase);
    # Model Initialize
    model = BertForTokenClassification.from_pretrained(opt.model, num_labels=opt.nLabels);
    
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(opt.loadModel))
    if opt.classType != "all_class":
        binary = True
    else:
        binary = False
    prop_tech_e, prop_tech, _, _, p2id = settings(opt.techniques, opt.binaryLabel, opt.bio)
    ids, texts, _ = read_data(opt.testDataset, isLabels = False)

    logging.info("Files loaded from directory: %s" % (len(texts)))

    flat_list_i, flat_list, flat_list_s = test2list(ids, texts)
    cleaned = [[tokenizer.tokenize(words) for words in sent] for sent in flat_list]
    tokenized_texts = [concatenate_list_data(sent) for sent in cleaned]
    print(tokenized_texts[:50])
    numerics = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              max_len=opt.maxLen)
    attention_masks = [[float(i>0) for i in ii] for ii in numerics]
    print (numerics[:5], attention_masks[:5])
    t_inputs = torch.tensor(numerics)
    t_masks = torch.tensor(attention_masks)

    t_data = TensorDataset(t_inputs, t_masks)
    t_dataloader = DataLoader(t_data, sampler=None, batch_size=opt.validBatch)

    model.eval()
    predictions_sample = []

    for batch in tqdm(t_dataloader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask= batch
        logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        predictions_sample.extend([list(p) for p in np.argmax(logits, axis=2)])

    df = get_char_level(flat_list_i, flat_list_s, predictions_sample, cleaned, hash_token, end_token, prop_tech)
    postfix = opt.testDataset.rsplit('/', 2)[-2]
    out_dir = opt.loadModel.rsplit('/', 1)[0] + "/pred." + postfix
    df.to_csv(out_dir, sep='\t', index=False, header=False) 
    logging.info("Predictions written to: %s" % (out_dir))

    out_file = opt.loadModel.rsplit('/', 1)[0] + "/score." + postfix
    if opt.classType != "binary":
        char_predict = tools.task3_scorer_onefile.main(["-s", out_dir, "-r", opt.testDataset, "-t", "./tools/data/propaganda-techniques-names.txt", "-l", out_file])
    else:
        char_predict = tools.task3_scorer_onefile.main(["-s", out_dir, "-r", opt.testDataset, "-t", "./tools/data/propaganda-techniques-names.txt", "-f", "-l", out_file])

    #text_file = open(out_dir, "w")
    #text_file.write(str(char_predict))
    #text_file.close()
    logging.info("Scores written to: %s" % (out_dir))
    print(prop_tech)

if __name__ == '__main__':
    main()
