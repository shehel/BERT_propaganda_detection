from tokenize_text import *
from early_stopping import EarlyStopping                            
from utils import *
from opt import opt

from datetime import timedelta, datetime
import pickle
import logging
import numpy as np
import pandas as pd
import torch
import pdb
import random
from tqdm import tqdm, trange
import os 
import itertools
import matplotlib.pyplot as plt 

from pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupCosineWithHardRestartsSchedule
from pytorch_transformers import *

from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as f1
from sklearn.model_selection import train_test_split

from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler, TensorDataset


# Derive sentence level scores
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
    plt.plot(x, trainlosses, label = "Train loss") 

    plt.plot(x, validlosses, label = "Validation losses") 

    plt.plot(x, f1scores, label = "F1 scores char level") 
    
    plt.plot(x, f1scores_word, label = "F1 scores word level") 
    plt.plot(x, task2_scores, label = "F1 scores task2") 

    plt.xlabel('Epochs') 
    plt.ylabel('Metric') 
    plt.title('Training Curves') 
    plt.legend() 
  
    plt.savefig("exp/{}/{}/learning_curves.png".format(opt.classType, opt.expID))
    
def main():
    MODEL_CLASSES = {
    'bert': (BertConfig, BertForTokenClassification, BertTokenizer, 'bert-base-uncased'),
    }
    _, model_class, tokenizer_class, pretrained_weights = MODEL_CLASSES[opt.model]

    # os.environ['CUDA_VISIBLE_DEVICES']='1'
    make_logger()
    prop_tech_e, prop_tech, hash_token, end_token, p2id = settings(opt.techniques, opt.binaryLabel, opt.bio)
    logging.info("Training for class %s" % (opt.binaryLabel))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count();

    # Seed everything
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # Additional seed
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    if n_gpu > 0:
        torch.cuda.manual_seed_all(opt.seed) 
    logging.info("GPUs Detected: %s" % (n_gpu)) 
    scorred_labels = list(range(1,(opt.nLabels-2)))
    
    if opt.loadModel:
        print('Loading Model from {}'.format(opt.loadModel))
        model = model_class.from_pretrained(opt.loadModel)
        # Loading the trained tokenizer sometimes cause poor results probably
        # due to mismatch between train and test set
        #tokenizer = tokenizer_class.from_pretrained(opt.loadModel, do_lower_case=opt.lowerCase)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=opt.lowerCase)

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
    
    model.to(device)
    
    print ("Auxiliary tokens: ", hash_token, end_token)

    # Load Tokenized train and validation datasets
    tr_inputs, tr_tags, tr_masks, _ = make_set(p2id, opt.trainDataset, tokenizer, opt.binaryLabel, hash_token, end_token)
    val_inputs, val_tags, val_masks, cleaned, flat_list_i, flat_list, flat_list_s,_ = make_val_set(p2id, opt.evalDataset,
                                                                                             tokenizer, opt.binaryLabel, hash_token, end_token)
    # True labels for sentence level predictions
    truth_task2 = get_task2(val_tags)
    
    logging.info("Dataset loaded")
    logging.info("Labels detected in train dataset: %s" % (np.unique(tr_tags)))
    logging.info("Labels detected in val dataset: %s" % (np.unique(val_tags)))

    # Balanced Sampling
    # total_tags = np.zeros((opt.nLabels,))
    # for x in tr_tags:
    #      total_tags = total_tags+np.bincount(x)
    
    # probs = 1./total_tags
    # train_tokenweights = probs[tr_tags]
    # weightage = np.sum(train_tokenweights, axis=1)
    
    # Uniformly sampling propaganda fragments more
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
    num_warmup_steps = 100

    num_train_optimization_steps = int(len(train_data) / opt.trainBatch ) * opt.nEpochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
     
    n_layers, n_heads = model.bert.config.num_hidden_layers, model.bert.config.num_attention_heads
    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.LR, correct_bias=False)

    scheduler = WarmupConstantSchedule(optimizer, warmup_steps = num_warmup_steps)
    #scheduler = WarmupCosineWithHardRestartsSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_train_optimization_steps, cycles=opt.nEpochs)

    if opt.fp16:
        logging.info("Model training in FP16")
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    #head_mask = torch.zeros(n_layers, n_heads).to(device)
    head_mask = None
    # head_mask[8:] = 0

    # Bertology
    if opt.bology:
        compute_heads_importance(device, model, valid_dataloader, compute_entropy=True,
                        compute_importance=True, head_mask=head_mask)

        aux = [flat_list_i, flat_list_s, cleaned, hash_token, end_token, prop_tech]
        #head_mask = mask_heads(device, model, valid_dataloader, aux)
        #prune_heads(device, model, valid_dataloader, head_mask, aux)
        return

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
                            attention_mask=b_input_mask, labels=b_labels, head_mask=head_mask)
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
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            logging.info(f'EPOCH {i} done: Train Loss {(tr_loss/nb_tr_steps)}')
            train_losses.append(tr_loss/nb_tr_steps)
       
        # Evaluation on validation set or test set
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        attn_weights = []
        for count, batch in enumerate(tqdm(valid_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            with torch.no_grad():
                tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask, labels=b_labels, head_mask=head_mask)
                #pdb.set_trace()
                #attn_weight = tmp_eval_loss[2]
                tmp_eval_loss = tmp_eval_loss[0]
                logits = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, head_mask=head_mask)
                logits = logits[0]
            #pdb.set_trace()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)

            #attn_weightx = tuple(t.detach().cpu().tolist() for t in attn_weight)
            b_input_ids, b_input_mask, b_labels = batch
            # ttn_weights.extend(attn_weightx)
            if count == 1:
                print (count)
            # break 

            #tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            
            eval_loss += tmp_eval_loss.mean().item()
            #eval_accuracy += tmp_eval_accuracy
            
            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
        # with open('attn.pp', 'wb') as fp:
        #     pickle.dump(attn_weight, fp)
        # with open('flatls', 'wb') as fp:
        #     pickle.dump(cleaned, fp)
        # break
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
        if not opt.train:
            break

        #     with open('predictions', 'wb') as fp:
        #         pickle.dump(predictions, fp)

        #     with open('val', 'wb') as fp:
        #         pickle.dump(val_tags, fp)

        #     with open('val_inp', 'wb') as fp:
        #         pickle.dump(cleaned, fp)
        #     #with open('val_mask', 'wb') as fp:
        #     #    pickle.dump(flat_list_s, fp)
            
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
