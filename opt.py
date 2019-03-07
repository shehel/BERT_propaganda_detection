
import argparse

parser = argparse.ArgumentParser(description='PyTorch BERT model Training')

"----------------------------- General options -----------------------------"
parser.add_argument('--expID', default='default', type=str,
                    help='Experiment ID')
parser.add_argument('--trainDataset', default='', type=str,
                    help='CSV file containing the training dataset')
parser.add_argument('--valDataset', default='', type=str,
                    help='CSV file containing the validation dataset')               
parser.add_argument('--snapshot', default=1, type=int,
                    help='How often to take a snapshot of the model (0 = never)')
parser.add_argument('--classType', default="all_class", type=str,
                    help='all_class | single_label | binary')
parser.add_argument('--outputFile', default="pred_local.csv", type=str,
                    help='Directory and name of the file to store predictions.')
"----------------------------- Model options -----------------------------"
parser.add_argument('--model', default="bert-large-cased", type=str,
                    help='Select a model to be trained: bert-base-cased|bert-base-uncased|bert-large-uncased|bert-large-cased')
parser.add_argument('--lowerCase', default=False, type=bool,
                    help='Set to true if using a uncased model')
parser.add_argument('--nLabels', default=21, type=int,
                    help='Number of labels to predict')
parser.add_argument('--loadModel', default=None, type=str,
                    help='Provide full path to a previously trained model')

parser.add_argument('--maxLen', default=210, type=float,
                    help='Max length of tokens in a single training sample')
"----------------------------- Hyperparameter options -----------------------------"
parser.add_argument('--LR', default=1e-3, type=float,
                    help='Learning rate')

parser.add_argument('--momentum', default=0, type=float,
                    help='Momentum')
parser.add_argument('--weightDecay', default=0, type=float,
                    help='Weight decay')



"----------------------------- Training options -----------------------------"
parser.add_argument('--nEpochs', default=10, type=int,
                    help='Number of hourglasses to stack')

parser.add_argument('--trainBatch', default=32, type=int,
                    help='Train-batch size')
parser.add_argument('--validBatch', default=32, type=int,
                    help='Valid-batch size')


opt = parser.parse_args()
