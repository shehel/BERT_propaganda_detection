import logging
from tokenize_text import *

import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import (BasicTokenizer, BertConfig,
                                     BertForTokenClassification, BertTokenizer)
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as f1
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler, TensorDataset
                            
from tqdm import tqdm, trange
import os 
from opt import opt
import itertools

logging.basicConfig(level=logging.INFO)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count(); 
    logging.info("GPUs Detected: %s" % (n_gpu))

    tokenizer = BertTokenizer.from_pretrained(opt.model, do_lower_case=opt.lowerCase);

    # Load Tokenized train and validation datasets
    trn_tokenized_texts, trn_label_l, tr_inputs, tr_tags, tr_masks = make_set(opt.trainDataset, tokenizer)
    val_tokenized_texts, val_label_l, val_inputs, val_tags, val_masks = make_set(opt.valDataset, tokenizer)
    logging.info("Dataset loaded")
    logging.info("Labels detected in train dataset: %s" % (np.unique(tr_tags)))
    logging.info("Labels detected in val dataset: %s" % (np.unique(val_tags)))


    # Balanced Sampling
    total_tags = np.zeros((21,))
    for x in tr_tags:
        total_tags = total_tags+np.bincount(x)
    
    probs = 1./total_tags
    train_tokenweights = probs[tr_tags]
    weightage = np.sum(train_tokenweights, axis=1)
    
    ## Convert to pyTorch tensors
    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_tags = torch.tensor(tr_tags)
    val_tags = torch.tensor(val_tags)
    tr_masks = torch.tensor(tr_masks)
    val_masks = torch.tensor(val_masks)

    ## Create Dataloaders
    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = WeightedRandomSampler(weights=weightage, num_samples=len(tr_tags),replacement=True)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=opt.trainBatch)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=opt.trainBatch)

    
    # Model Initialize
    model = BertForTokenClassification.from_pretrained(opt.model, num_labels=opt.nLabels);
    if opt.loadModel:
        print('Loading Model from {}'.format(opt.loadModel))
        m.load_state_dict(torch.load(opt.loadModel))
        if not os.path.exists("../exp/{}/{}".format(opt.classType, opt.expID)):
            try:
                os.mkdir("./exp/{}/{}".format(opt.classType, opt.expID))
            except FileNotFoundError:
                os.mkdir("./exp/{}".format(opt.classType))
                os.mkdir("./exp/{}/{}".format(opt.classType, opt.expID))
    else:
        print('Create new model')
        if not os.path.exists("./exp/{}/{}".format(opt.classType, opt.expID)):
            try:
                os.mkdir("./exp/{}/{}".format(opt.classType, opt.expID))
            except FileNotFoundError:
                os.mkdir("./exp/{}".format(opt.classType))
                os.mkdir("./exp/{}/{}".format(opt.classType, opt.expID))

    loss_scale = 0
    warmup_proportion = 0.1
    num_train_optimization_steps = int(len(train_data) / opt.trainBatch ) * opt.nEpochs
    
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=opt.LR,
                         warmup=warmup_proportion,
                         t_total=-1,
                         max_grad_norm=1.0)    
    
    model.to(device)
    
    print("Training beginning on: ", n_gpu)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # F1 score shouldn't consider no-propaganda
    # and other auxiliary labels
    if (opt.classType == "all_class"):
        scorred_labels = list(range(1, 19))
    else:
        scorred_labels = [1]

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    max_grad_norm = 1.0
    best = 0
    for _ in trange(opt.nEpochs, desc="Epoch"):
        # TRAIN loop
        #scheduler.step()
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # forward pass
            loss = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
            if n_gpu > 1:
                loss = loss.mean()
            
            # backward pass
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            
            #torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        # print train loss per epoch
        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        # VALIDATION on validation set
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        for batch in tqdm(valid_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            with torch.no_grad():
                tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask, labels=b_labels)
                logits = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)
            
            #tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            
            eval_loss += tmp_eval_loss.mean().item()
            #eval_accuracy += tmp_eval_accuracy
            
            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))    
        print("Precision, Recall, F1-Score, Support: {}".format(f1(list(itertools.chain(*predictions)), list(itertools.chain(*val_tags)), average=None)))
        f1_macro = f1_score(list(itertools.chain(*predictions)), list(itertools.chain(*val_tags)), labels=scorred_labels, average="macro")
        print("F1 Macro ", f1_macro)
        #scheduler.step(f1_macro)
        print(optimizer.get_lr()[0])
        
        #TODO save checkpoints
        #Save model based on best F1 score
        if f1_macro > best:
        # Save a trained model and the associated configuration
        #TODO store configuration file
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join("./exp/{}/{}".format(opt.classType, opt.expID), "best_model.pth")
            torch.save(model_to_save.state_dict(), output_model_file)
            best = f1_macro
            logging.info("New best model")

    logging.info("Training Finished")

if __name__ == '__main__':
    main()
