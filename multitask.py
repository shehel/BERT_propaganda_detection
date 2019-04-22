from pytorch_pretrained_bert.modeling import BertPreTrainedModel

class BertForMultiTask(BertPreTrainedModel):
    """BERT model for multi-task learning.
    This module is composed of the BERT model with two linear layers on top 
    of a BERT model.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels_token`: labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [0, ..., num_labels].
        'labels_sequence': labels for the sequence classification outrput.
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels_t, num_labels_s):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels_t = num_labels_t
        self.num_labels_s = num_labels_s
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_t = nn.Linear(config.hidden_size, num_labels_t)
        self.classifier_s = nn.Linear(config.hidden_size, num_labels_s)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels_t=None, labels_s=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits_t = self.classifier_t(sequence_output)
        logits_s = self.classifier_s(pooled_output)
        ret_t = None
        ret_s = None
        if labels_t is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits_t.view(-1, self.num_labels_t)[active_loss]
                active_labels = labels_t.view(-1)[active_loss]
                loss_t = loss_fct(active_logits, active_labels)
            else:
                loss_t = loss_fct(logits_t.view(-1, self.num_labels_t), labels_t.view(-1))
            loss_s = loss_fct(logits_s.view(-1, self.num_labels_s), labels_s.view(-1))
            return loss_t, loss_s
        else:
            return logits_t, logits_s

