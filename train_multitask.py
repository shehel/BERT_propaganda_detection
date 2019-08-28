import pickle
import logging
from tokenize_text import *
from utils import *
import numpy as np
import pandas as pd
import torch
import pdb

import random
from random import randint
#from pytorch_pretrained_bert import (BasicTokenizer, BertConfig,
#                                     BertForTokenClassification, BertTokenizer)
#from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from models import XLNetForTokenClassification, GPT2ForTokenClassification
from pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupCosineWithHardRestartsSchedule
from pytorch_transformers import *

from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as f1
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler, TensorDataset
from early_stopping import EarlyStopping                            
from tqdm import tqdm, trange
import os 
from opt import opt
import itertools
import os

import matplotlib.pyplot as plt 



def make_logger() -> None:
    if not os.path.exists("./exp/{}/{}".format(opt.classType, opt.expID)):
            try:
                os.mkdir("./exp/{}/{}".format(opt.classType, opt.expID))
            except FileNotFoundError:
                os.mkdir("./exp/{}".format(opt.classType))
                os.mkdir("./exp/{}/{}".format(opt.classType, opt.expID))
    
    logging.basicConfig(
    filename= ("./exp/{}/{}/log.txt".format(opt.classType, opt.expID)),
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s, %(message)s')

    logging.getLogger().addHandler(logging.StreamHandler())

def draw_curves(trainlosses, validlosses, f1scores, f1scores_word, task2_scores) -> None:
    x = list(range(len(validlosses)))
    # plotting the line 1 points  
    plt.plot(x, trainlosses, label = "Train loss") 

    # plotting the line 2 points  
    plt.plot(x, validlosses, label = "Validation losses") 

    # line 3 points 
    # plotting the line 2 points  
    plt.plot(x, f1scores, label = "F1 scores char level") 
    
    plt.plot(x, f1scores_word, label = "F1 scores word level") 
    plt.plot(x, task2_scores, label = "F1 scores task2") 

    plt.xlabel('Epochs') 
    plt.ylabel('Metric') 
    plt.title('Training Curves') 
    #plt.yscale('log')
    # show a legend on the plot 
    plt.legend() 
  
    plt.savefig("exp/{}/{}/learning_curves.png".format(opt.classType, opt.expID))
    
def main():
    MODEL_CLASSES = {
    'robert':(RobertaConfig, roBertForTokenClassification,  RobertaTokenizer, 'roberta-base'),
    'bert': (BertConfig, BertForMultitask, BertTokenizer, 'bert-base-uncased'),
    'xlnet': (XLNetConfig, XLNetForTokenClassification, XLNetTokenizer, 'xlnet-base-cased'),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer, 'xlm-mlm-en-2048')
    }
    _, model_class, tokenizer_class, pretrained_weights = MODEL_CLASSES[opt.model]

    #os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
    make_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count();
    
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(opt.seed)

    logging.info("GPUs Detected: %s" % (n_gpu))

    logging.info("Setting up dataloader")
    rtasks = ["Name_Calling,Labeling"]

    tasks = ["Appeal_to_Authority", "Appeal_to_fear-prejudice", "Bandwagon", "Black-and-White_Fallacy",
             "Causal_Oversimplification", "Doubt", "Exaggeration,Minimisation", "Flag-Waving", "Loaded_Language",
             "Name_Calling,Labeling", "Obfuscation,Intentional_Vagueness,Confusion", "Red_Herring", "Reductio_ad_hitlerum",
             "Repetition", "Slogans", "Straw_Men", "Thought-terminating_Cliches", "Whataboutism"]
    num_tasks = len(tasks)
    if opt.loadModel:
        print('Loading Model from {}'.format(opt.loadModel))
        model = model_class.from_pretrained(opt.loadModel, tasks = num_tasks)
        #tokenizer = tokenizer_class.from_pretrained(opt.loadModel, do_lower_case=opt.lowerCase)

        #model.load_state_dict(torch.load(opt.loadModel))
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=opt.lowerCase)
        if not os.path.exists("./exp/{}/{}".format(opt.classType, opt.expID)):
            try:
                os.mkdir("./exp/{}/{}".format(opt.classType, opt.expID))
            except FileNotFoundError:
                os.mkdir("./exp/{}".format(opt.classType))
                os.mkdir("./exp/{}/{}".format(opt.classType, opt.expID))
    else:
        model = model_class.from_pretrained(pretrained_weights, num_labels=opt.nLabels, tasks=num_tasks)
        
        #tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=opt.lowerCase)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=opt.lowerCase)

        print('Create new model')
        if not os.path.exists("./exp/{}/{}".format(opt.classType, opt.expID)):
            try:
                os.mkdir("./exp/{}/{}".format(opt.classType, opt.expID))
            except FileNotFoundError:
                os.mkdir("./exp/{}".format(opt.classType))
                os.mkdir("./exp/{}/{}".format(opt.classType, opt.expID))
    
    model.to(device)

    ## Creating dataloaders
    train_postfix = '.train'
    dev_postfix = '.dev'
    tr_loaders = []
    dev_loaders = []
    prop_techs= []
    for i, task_name in enumerate(tasks):
        prop_tech_e, prop_tech, hash_token, end_token, p2id = settings(opt.techniques, task_name, opt.bio)
        prop_techs.append(prop_tech)
        scorred_labels = list(range(1,(opt.nLabels-2)))
        
        print (hash_token, end_token)
        # Load Tokenized train and validation datasets
        tr_inputs, tr_tags, tr_masks, _ = make_set(p2id, opt.trainDataset+task_name+train_postfix, tokenizer, task_name, hash_token, end_token)
        val_inputs, val_tags, val_masks, cleaned, flat_list_i, flat_list, flat_list_s,_ = make_val_set(p2id, opt.evalDataset+task_name+dev_postfix,
                                                                                                tokenizer, task_name, hash_token, end_token)
        logging.info("Dataset loaded")
        logging.info("Labels detected in train dataset: %s" % (np.unique(tr_tags)))
        logging.info("Labels detected in val dataset: %s" % (np.unique(val_tags)))

        # Balanced Sampling
        total_tags = np.zeros((opt.nLabels,))
        for x in tr_tags:
            total_tags = total_tags+np.bincount(x)
        
        probs = 1./total_tags
        train_tokenweights = probs[tr_tags]
        weightage = np.sum(train_tokenweights, axis=1)

        # Alternate method for weighting
        ws = np.ones((opt.nLabels,))
        ws[0] = 0
        
        ws[hash_token] = 0
        ws[end_token] = 0
        ws = ws+0.3
        prob = [max(x) for x in ws[tr_tags]]
        weightage = [x + y for x, y in zip(prob, (len(prob)*[0.1]))]    
        
        # Convert to pyTorch tensors
        tr_inputs = torch.tensor(tr_inputs)
        val_inputs = torch.tensor(val_inputs)
        tr_tags = torch.tensor(tr_tags)
        val_tags = torch.tensor(val_tags)
        tr_masks = torch.tensor(tr_masks)
        val_masks = torch.tensor(val_masks)
        
        # Create Dataloaders
        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        #train_sampler = WeightedRandomSampler(weights=weightage, num_samples=len(tr_tags),replacement=False)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=opt.trainBatch)

        valid_data = TensorDataset(val_inputs, val_masks, val_tags)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=opt.trainBatch)

        tr_loaders.append(train_dataloader)
        dev_loaders.append(valid_dataloader)
    
    
    max_grad_norm = 1.0
    num_total_steps = 113
    num_warmup_steps = 100
    warmup_proportion = float(num_warmup_steps) / float(num_total_steps)
    #loss_scale = 0
    #warmup_proportion = 0.1

    num_train_optimization_steps = 1000 * opt.nEpochs
    t_total = len(train_data) // opt.nEpochs
    print ("t_total______________________________", t_total)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.LR, correct_bias=False)
    #optimizer = torch.optim.SGD(mqodel.parameters(), lr=opt.LR, momentum=0.9)
    scheduler = WarmupCosineWithHardRestartsSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_train_optimization_steps, cycles=opt.nEpochs)
    if opt.fp16:
        logging.info("Model training in FP16")
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        logging.info("Training beginning on: %s" % n_gpu)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    max_grad_norm = 1.0
    best = 0
    early_stopping = EarlyStopping(patience=opt.patience, verbose=True)
    train_losses = []
    valid_losses = []
    f1_scores = []
    f1_scores_word = []
    task2_scores = []
    
    for i in trange(opt.nEpochs, desc="Epoch"):
        if (opt.train):
            model.train()
            tr_loader_iters = [iter(train_loader) for train_loader in tr_loaders]
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step in trange(1000, desc="Iterations"):
            #for batch in tqdm(tr_loaders[0]):
            # TRAIN loop
            # Start only if train flag was passed
                k = randint(0, num_tasks-1)
                batch = next(tr_loader_iters[k])  
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                
                # forward pass
                loss = model(b_input_ids, k, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
                loss = loss[0]
                if n_gpu > 1:
                    loss = loss.mean()

                # backward pass
                if opt.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            logging.info(f'EPOCH {i} done: Train Loss {(tr_loss/nb_tr_steps)}')
            train_losses.append(tr_loss/nb_tr_steps)
       
        # Evaluation on validation set or test set
        master_df = pd.DataFrame()
        model.eval()
        
        for l, valid_dataloader in enumerate(tqdm(dev_loaders, desc="Evaluating")):
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            predictions , true_labels = [], []
            for batch in tqdm(valid_dataloader, desc="Evaluating"):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                
                with torch.no_grad():
                    tmp_eval_loss = model(b_input_ids, l, token_type_ids=None,
                                        attention_mask=b_input_mask, labels=b_labels)
                    tmp_eval_loss = tmp_eval_loss[0]
                    logits = model(b_input_ids, l, token_type_ids=None,
                                attention_mask=b_input_mask)
                    logits = logits[0]
                #pdb.set_trace()
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.append(label_ids)
                
                
                eval_loss += tmp_eval_loss.mean().item()
                
                nb_eval_examples += b_input_ids.size(0)
                nb_eval_steps += 1

            if i % opt.snapshot == 0:
                if not os.path.exists("./exp/{}/{}/{}".format(opt.classType, opt.expID, i)):
                    try:
                        os.mkdir("./exp/{}/{}/{}".format(opt.classType, opt.expID, i))
                    except FileNotFoundError:
                        os.mkdir("./exp/{}/{}/{}".format(opt.classType, opt.expID, i))
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained('./exp/{}/{}/{}/'.format(opt.classType, opt.expID, i))
                tokenizer.save_pretrained('./exp/{}/{}/{}/'.format(opt.classType, opt.expID, i))
 
            eval_loss = eval_loss/nb_eval_steps
            logging.info("Validation loss: %s" % (eval_loss))    
            logging.info("Precision, Recall, F1-Score, Support: {}".format(f1(list(itertools.chain(*predictions)), list(itertools.chain(*val_tags)), average=None)))
            f1_macro = f1_score(list(itertools.chain(*predictions)), list(itertools.chain(*val_tags)), labels=scorred_labels, average="macro")
            logging.info("F1 Macro Dev Set: %s" % f1_macro)
            valid_losses.append(eval_loss)
            f1_scores_word.append(f1_macro)
            prop_tech = prop_techs[l] 
            df = get_char_level(flat_list_i, flat_list_s, predictions, cleaned, hash_token, end_token, prop_tech)
            if not df.empty:
                master_df = master_df.append(df)
            postfix = opt.testDataset.rsplit('/', 2)[-2]
            if opt.loadModel:
                out_dir = opt.loadModel.rsplit('/', 1)[0] + "/pred." + postfix
            else:
                out_dir = ("exp/{}/{}/temp_pred{}.csv".format(opt.classType, opt.expID, l))
            df.to_csv(out_dir, sep='\t', index=False, header=False) 
            logging.info("Predictions written to: %s" % (out_dir))

            #if opt.loadModel:
            #    out_file = opt.loadModel.rsplit('/', 1)[0] + "/score." + postfix
            #else:
        if opt.loadModel:
            out_dir = opt.loadModel.rsplit('/', 1)[0] + "/pred." + postfix
        else:
            out_dir = ("exp/{}/{}/temp_pred_consol{}.csv".format(opt.classType, opt.expID, l))

        master_df.to_csv(out_dir, sep='\t', index=False, header=False) 
        logging.info("Consolidated predictions written to: %s" % (out_dir))
        if opt.loadModel:
            out_file = opt.loadModel.rsplit('/', 1)[0] + "/score." + postfix
        else:
            out_file = ("exp/{}/{}/temp_score_consol.csv".format(opt.classType, opt.expID))


        if opt.classType == "binary":
            char_predict = tools.task3_scorer_onefile.main(["-s", out_dir, "-r", opt.testDataset, "-t", opt.techniques, "-l", out_file])
        else:
            #char_predict = tools.task3_scorer_onefile.main(["-s", out_dir, "-r", opt.testDataset, "-t", opt.techniques, "-f", "-l", out_file])
            a = os.popen("python tools/task-FLC_scorer.py -s " + out_dir+ " -r "+opt.testDataset).read()
            char_predict = float(a.split("F1=")[1].split("\n")[0])
            logging.info(a)    
        f1_scores.append(char_predict) 
        print (char_predict)
            
        if not opt.train:
            break   

        early_stopping(char_predict*(-1), model, tokenizer)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break
        
if __name__ == '__main__':
    main()
