import numpy as np
import torch
from opt import opt
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, tokenizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, tokenizer)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, tokenizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, tokenizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model_to_save = model.module if hasattr(model, 'module') else model  
        model_to_save.save_pretrained('./exp/{}/{}/'.format(opt.classType, opt.expID))
        tokenizer.save_pretrained('./exp/{}/{}/'.format(opt.classType, opt.expID))

        #torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
