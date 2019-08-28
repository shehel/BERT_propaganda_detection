import pickle
import logging
from tokenize_text import *
import tools_v0.task3_scorer_onefile
from utils import *
import numpy as np
import pandas as pd
import torch
import pdb

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

def get_task2(predictions):
    preddi = []
    found = False
    for x in predictions:
        for j in x:
            if j==1:
                preddi.append(1)
                found = True
                break
        if not found:
            preddi.append(0)
        found = False
    return preddi

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

    # naming the x axis 
    plt.xlabel('Epochs') 
    # naming the y axis 
    plt.ylabel('Metric') 
    # giving a title to my graph 
    plt.title('Training Curves') 
    #plt.yscale('log')
    # show a legend on the plot 
    plt.legend() 
  
    plt.savefig("exp/{}/{}/learning_curves.png".format(opt.classType, opt.expID))
    
def main():
    MODEL_CLASSES = {
    'robert':(RobertaConfig, roBertForTokenClassification,  RobertaTokenizer, 'roberta-base'),
    'bert': (BertConfig, BertForTokenClassification, BertTokenizer, 'bert-base-uncased'),
    'xlnet': (XLNetConfig, XLNetForTokenClassification, XLNetTokenizer, 'xlnet-base-cased'),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer, 'xlm-mlm-en-2048')
    }
    _, model_class, tokenizer_class, pretrained_weights = MODEL_CLASSES[opt.model]

    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
    make_logger()
    prop_tech_e, prop_tech, hash_token, end_token, p2id = settings(opt.techniques, opt.binaryLabel, opt.bio)
    logging.info("Training for class %s" % (opt.binaryLabel))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count();
    
    

    #random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(opt.seed)
        
    logging.info("GPUs Detected: %s" % (n_gpu))
    scorred_labels = list(range(1,(opt.nLabels-2)))
    #pdb.set_trace()
    if opt.loadModel:
        print('Loading Model from {}'.format(opt.loadModel))
        model = model_class.from_pretrained(opt.loadModel)
        #tokenizer = tokenizer_class.from_pretrained(opt.loadModel, do_lower_case=opt.lowerCase)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=opt.lowerCase)

        #model.load_state_dict(torch.load(opt.loadModel))
        if not os.path.exists("./exp/{}/{}".format(opt.classType, opt.expID)):
            try:
                os.mkdir("./exp/{}/{}".format(opt.classType, opt.expID))
            except FileNotFoundError:
                os.mkdir("./exp/{}".format(opt.classType))
                os.mkdir("./exp/{}/{}".format(opt.classType, opt.expID))
    else:
        

        model = model_class.from_pretrained(pretrained_weights, num_labels=opt.nLabels)
        
        #tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=opt.lowerCase)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=opt.lowerCase)

        print('Create new model')
        if not os.path.exists("./exp/{}/{}".format(opt.classType, opt.expID)):
            try:
                os.mkdir("./exp/{}/{}".format(opt.classType, opt.expID))
            except FileNotFoundError:
                os.mkdir("./exp/{}".format(opt.classType))
                os.mkdir("./exp/{}/{}".format(opt.classType, opt.expID))
    # Model Initialize
    
    model.to(device)
    
    print (hash_token, end_token)
    # Load Tokenized train and validation datasets
    tr_inputs, tr_tags, tr_masks, _ = make_set(p2id, opt.trainDataset, tokenizer, opt.binaryLabel, hash_token, end_token)
    val_inputs, val_tags, val_masks, cleaned, flat_list_i, flat_list, flat_list_s,_ = make_val_set(p2id, opt.evalDataset,
                                                                                             tokenizer, opt.binaryLabel, hash_token, end_token)
    printable = tr_tags
    # ids, texts, _ = read_data(opt.testDataset, isLabels = False)
    # flat_list_i, flat_list, flat_list_s = test2list(ids, texts)
    truth_task2 = get_task2(val_tags)
    
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
    train_sampler = WeightedRandomSampler(weights=weightage, num_samples=len(tr_tags),replacement=True)
    #train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=opt.trainBatch)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=opt.trainBatch)
    
    max_grad_norm = 1.0
    num_total_steps = 225
    num_warmup_steps = 100
    warmup_proportion = float(num_warmup_steps) / float(num_total_steps)
    #loss_scale = 0
    #warmup_proportion = 0.1

    num_train_optimization_steps = int(len(train_data) / opt.trainBatch ) * opt.nEpochs
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
    scheduler = WarmupConstantSchedule(optimizer, warmup_steps = num_warmup_steps)
    #scheduler = WarmupCosineWithHardRestartsSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_train_optimization_steps, cycles=opt.nEpochs)
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

    # F1 score shouldn't consider no-propaganda
    # and other auxiliary labels
    #pdb.set_trace()
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
    best = 0
    for i in trange(opt.nEpochs, desc="Epoch"):
        # TRAIN loop
        # Start only if train flag was passed
        if (opt.train):
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
                tmp_eval_loss = tmp_eval_loss[0]
                logits = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)
                logits = logits[0]
            #pdb.set_trace()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)
            
            #tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            
            eval_loss += tmp_eval_loss.mean().item()
            #eval_accuracy += tmp_eval_accuracy
            
            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
        if i % opt.snapshot == 0:
            if not os.path.exists("./exp/{}/{}/{}".format(opt.classType, opt.expID, i)):
                try:
                    os.mkdir("./exp/{}/{}/{}".format(opt.classType, opt.expID, i))
                except FileNotFoundError:
                    os.mkdir("./exp/{}/{}/{}".format(opt.classType, opt.expID, i))
            #torch.save(
            #    model.state_dict(), './exp/{}/{}/{}/model_{}.pth'.format(opt.classType, opt.expID, i, i))
            #torch.save(
            #    opt, './exp/{}/{}/{}/option.pth'.format(opt.classType, opt.expID, i))
            #torch.save(
            #    optimizer, './exp/{}/{}/{}/optimizer.pth'.format(opt.classType, opt.expID, i))
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained('./exp/{}/{}/{}/'.format(opt.classType, opt.expID, i))
            tokenizer.save_pretrained('./exp/{}/{}/{}/'.format(opt.classType, opt.expID, i))
 
        pred_task2 = get_task2(predictions)
        logging.info("Precision, Recall, F1-Score, Support Task2: {}".format(f1(pred_task2, truth_task2, average=None)))
        f1_macro = f1_score(pred_task2, truth_task2, labels=scorred_labels, average="macro")
        task2_scores.append(f1_macro)
        pickle.dump(printable, open( "output_.p", "wb"))
        eval_loss = eval_loss/nb_eval_steps
        logging.info("Validation loss: %s" % (eval_loss))    
        logging.info("Precision, Recall, F1-Score, Support: {}".format(f1(list(itertools.chain(*predictions)), list(itertools.chain(*val_tags)), average=None)))
        f1_macro = f1_score(list(itertools.chain(*predictions)), list(itertools.chain(*val_tags)), labels=scorred_labels, average="macro")
        logging.info("F1 Macro Dev Set: %s" % f1_macro)
        #logging.info("Learning Rate: %s" % (optimizer.get_lr()[0]))
        valid_losses.append(eval_loss)
        f1_scores_word.append(f1_macro)
        
        df = get_char_level(flat_list_i, flat_list_s, predictions, cleaned, hash_token, end_token, prop_tech)
        postfix = opt.testDataset.rsplit('/', 2)[-2]
        if opt.loadModel:
            out_dir = opt.loadModel.rsplit('/', 1)[0] + "/pred." + postfix
        else:
            out_dir = ("exp/{}/{}/{}/temp_pred.csv".format(opt.classType, opt.expID, i))
        df.to_csv(out_dir, sep='\t', index=False, header=False) 
        logging.info("Predictions written to: %s" % (out_dir))

        if opt.loadModel:
            out_file = opt.loadModel.rsplit('/', 1)[0] + "/score." + postfix
        else:
            out_file = ("exp/{}/{}/temp_score.csv".format(opt.classType, opt.expID))

        if opt.classType == "binary":
            char_predict = tools.task3_scorer_onefile.main(["-s", out_dir, "-r", opt.testDataset, "-t", opt.techniques, "-l", out_file])
        else:
            #char_predict = tools.task3_scorer_onefile.main(["-s", out_dir, "-r", opt.testDataset, "-t", opt.techniques, "-f", "-l", out_file])
            a = os.popen("python tools/task-FLC_scorer.py -s " + out_dir+ " -r "+opt.testDataset).read()
            char_predict = float(a.split("F1=")[1].split("\n")[0])
            logging.info(a)    
        f1_scores.append(char_predict) 
        print (char_predict)
        if char_predict > best:
            best = char_predict 
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        # if not opt.train:

        #     with open('predictions', 'wb') as fp:
        #         pickle.dump(predictions, fp)

        #     with open('val', 'wb') as fp:
        #         pickle.dump(val_tags, fp)

        #     with open('val_inp', 'wb') as fp:
        #         pickle.dump(cleaned, fp)
        #     #with open('val_mask', 'wb') as fp:
        #     #    pickle.dump(flat_list_s, fp)
            
        #     break
        # with open('predictions_train', 'wb') as fp:
        #         pickle.dump(predictions, fp)
        early_stopping(char_predict*(-1), model, tokenizer)
        
        if early_stopping.early_stop:
            logging.info("Early stopping")
            print ("Best Score: ",best)
            break
        # Save checkpoints
       
        # Save model based on best F1 score and if epoch is greater than 3
        '''if f1_macro > best and i > 3:
        # Save a trained model and the associated configuration
            torch.save(
                model.state_dict(), './exp/{}/{}/best_model.pth'.format(opt.classType, opt.expID))
            torch.save(
                opt, './exp/{}/{}/option.pth'.format(opt.classType, opt.expID))
            torch.save(
                optimizer, './exp/{}/{}/optimizer.pth'.format(opt.classType, opt.expID))
            #model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            #output_model_file = os.path.join("./exp/{}/{}".format(opt.classType, opt.expID), "best_model.pth")
            #torch.save(model_to_save.state_dict(), output_model_file)
            best = f1_macro
            logging.info("New best model")
        '''
    if opt.train:
        print ("Best Score: ",best)
        logging.info("Training Finished. Learning curves saved.")
        draw_curves(train_losses, valid_losses, f1_scores, f1_scores_word, task2_scores)
        #df = pd.DataFrame({'col':trainlosses})
        #df.to_csv("trainlosses.csv", sep='\t', index=False, header=False) 
        #df = pd.DataFrame({'col':validlosses})
        #df.to_csv("validlosses.csv", sep='\t', index=False, header=False) 
        #df = pd.DataFrame({'col':f1scores})
        #df.to_csv("f1scores.csv", sep='\t', index=False, header=False) 
if __name__ == '__main__':
    main()
