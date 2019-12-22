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
