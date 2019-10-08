from tokenize_text import *
import tools_v0.task3_scorer_onefile
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

# Experimental
from models import XLNetForTokenClassification, GPT2ForTokenClassification

from pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupCosineWithHardRestartsSchedule
from pytorch_transformers import *

from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as f1
from sklearn.model_selection import train_test_split

from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler, TensorDataset

def entropy(p):
    """ Compute the entropy of a probability distribution """
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)

def print_2d_tensor(tensor):
    """ Print a 2D tensor """
    print("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            print(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            print(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))

def compute_heads_importance(device, model, eval_dataloader, compute_entropy=True,compute_importance=False, head_mask=None):
    n_layers, n_heads = model.bert.config.num_hidden_layers, model.bert.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(device)
    attn_entropy = torch.zeros(n_layers, n_heads).to(device)
    ## TODO: change args shit above

    dont_normalize_global_importance=False
    dont_normalize_importance_by_layer=True
    if head_mask is None:
        head_mask = torch.ones(n_layers, n_heads).to(device)
    if compute_importance:
        head_mask.requires_grad_(requires_grad=True)
    preds = None
    labels = None
    tot_tokens = 0.0
    preds, labels = [], []
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids = batch

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=label_ids, head_mask=head_mask)
        loss, logits, all_attentions = outputs[0], outputs[1], outputs[-1]  # Loss and logits are the first, attention the last
        loss.backward()  # Backpropagate to populate the gradients in the head mask

        if compute_entropy:
            for layer, attn in enumerate(all_attentions):
                #pdb.set_trace()
                masked_entropy = entropy(attn.detach()) * input_mask.float().unsqueeze(1)
                attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

        if compute_importance:
            head_importance += head_mask.grad.abs().detach()

        # Also store our logits/labels if we want to compute metrics afterwards
        # if preds is None:
        #     preds = logits.detach().cpu().numpy()
        #     labels = label_ids.detach().cpu().numpy()
        # else:
        #     preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        #     labels = np.append(labels, label_ids.detach().cpu().numpy(), axis=0)
        logits_d = logits.detach().cpu().numpy()
        label_d = label_ids.detach().cpu().numpy()
        preds.extend([list(p) for p in np.argmax(logits_d, axis=2)])
        labels.append(label_d)

        tot_tokens += input_mask.float().detach().sum().data

    # Normalize
    attn_entropy /= tot_tokens
    head_importance /= tot_tokens
    # Layerwise importance normalization
    if not dont_normalize_importance_by_layer:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1/exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    if dont_normalize_global_importance:
        head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())

    # Print/save matrices
    np.save(os.path.join('exp/all_class',opt.expID, 'attn_entropy.npy'), attn_entropy.detach().cpu().numpy())
    np.save(os.path.join('exp/all_class', opt.expID, 'head_importance.npy'), head_importance.detach().cpu().numpy())

    print("Attention entropies")
    print_2d_tensor(attn_entropy)
    print("Head importance scores")
    print_2d_tensor(head_importance)
    print("Head ranked by importance scores")
    head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=device)
    head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(head_importance.numel(), device=device)
    head_ranks = head_ranks.view_as(head_importance)
    print_2d_tensor(head_ranks)

    return attn_entropy, head_importance, preds, labels

def compute_metrics(predictions, aux=[]):
    #pdb.set_trace()
    df = get_char_level(aux[0], aux[1], predictions, aux[2], aux[3], aux[4], aux[5])
    postfix = opt.testDataset.rsplit('/', 2)[-2]
    if opt.loadModel:
        out_dir = opt.loadModel.rsplit('/', 1)[0] + "/pred." + postfix
    else:
        out_dir = ("exp/{}/{}/{}/temp_pred.csv".format(opt.classType, opt.expID, i))
    df.to_csv(out_dir, sep='\t', index=False, header=False) 
    print("Predictions written to: %s" % (out_dir))

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
        print(a)    
    
    print (char_predict)
    return char_predict

def mask_heads(device, model, eval_dataloader, aux=[]):
    """ This method shows how to mask head (set some heads to zero), to test the effect on the network,
        based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    masking_amount = 0.25
    masking_threshold = 0.9
    
    _, head_importance, preds, labels = compute_heads_importance(device, model, eval_dataloader, compute_entropy=False)
    # preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
    original_score = compute_metrics(preds, aux)
    print("Pruning: original score: %f, threshold: %f", original_score, original_score * masking_threshold)

    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = max(1, int(new_head_mask.numel() * masking_amount))

    current_score = original_score
    while current_score >= original_score * masking_threshold:
        head_mask = new_head_mask.clone() # save current head mask
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float('Inf')
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        if len(current_heads_to_mask) <= num_to_mask:
            break

        # mask heads
        current_heads_to_mask = current_heads_to_mask[:num_to_mask]
        print("Heads to mask: %s", str(current_heads_to_mask.tolist()))
        new_head_mask = new_head_mask.view(-1)
        new_head_mask[current_heads_to_mask] = 0.0
        new_head_mask = new_head_mask.view_as(head_mask)
        print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        _, head_importance, preds, labels = compute_heads_importance(device, model, eval_dataloader, compute_entropy=False, head_mask=new_head_mask)
        current_score = compute_metrics(preds, aux)
        print("Masking: current score: %f, remaning heads %d (%.1f percents)", current_score, new_head_mask.sum(), new_head_mask.sum()/new_head_mask.numel() * 100)

    print("Final head mask")
    print_2d_tensor(head_mask)
    np.save(os.path.join('exp/all_class',opt.expID, 'head_mask.npy'), head_mask.detach().cpu().numpy())

    return head_mask


def prune_heads(device, model, eval_dataloader, head_mask, aux=[]):
    """ This method shows how to prune head (remove heads weights) based on
        the head importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    # Try pruning and test time speedup
    # Pruning is like masking but we actually remove the masked weights
    before_time = datetime.now()
    _, _, preds, labels = compute_heads_importance(device, model, eval_dataloader,
                                                   compute_entropy=False, compute_importance=False, head_mask=head_mask)
    score_masking = compute_metrics(preds, aux)
    original_time = datetime.now() - before_time

    original_num_params = sum(p.numel() for p in model.parameters())
    heads_to_prune = dict((layer, (1 - head_mask[layer].long()).nonzero().tolist()) for layer in range(len(head_mask)))
    assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
    
    model.prune_heads(heads_to_prune)
    pruned_num_params = sum(p.numel() for p in model.parameters())

    before_time = datetime.now()
    _, _, preds, labels = compute_heads_importance(device, model, eval_dataloader,
                                                    compute_entropy=False, compute_importance=False, head_mask=None)
    
    score_pruning = compute_metrics(preds, aux)
    new_time = datetime.now() - before_time

    print("Pruning: original num of params: %.2e, after pruning %.2e (%.1f percents)", original_num_params, pruned_num_params, pruned_num_params/original_num_params * 100)
    print("Pruning: score with masking: %f score with pruning: %f", score_masking, score_pruning)
    print("Pruning: speed ratio (new timing / original timing): %f percents", original_time/new_time * 100)

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
    'robert':(RobertaConfig, roBertForTokenClassification,  RobertaTokenizer, 'roberta-base'),
    'bert': (BertConfig, BertForTokenClassification, BertTokenizer, 'bert-base-uncased'),
    'xlnet': (XLNetConfig, XLNetForTokenClassification, XLNetTokenizer, 'xlnet-base-cased'),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer, 'xlm-mlm-en-2048')
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
    
    print (hash_token, end_token)

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
    ws[hash_token] = 0
    ws[end_token] = 0
    ws = ws+0.5
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

    num_train_optimization_steps = int(len(train_data) / opt.trainBatch ) * opt.nEpochs
    t_total = len(train_data) // opt.nEpochs
    print ("t_total______________________________", t_total)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
     
    n_layers, n_heads = model.bert.config.num_hidden_layers, model.bert.config.num_attention_heads
    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.LR, correct_bias=False)

    scheduler = WarmupConstantSchedule(optimizer, warmup_steps = num_warmup_steps)
    # scheduler = WarmupCosineWithHardRestartsSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_train_optimization_steps, cycles=opt.nEpochs)

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
        attn_weights = []
        for count, batch in tqdm(enumerate(valid_dataloader)):
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
