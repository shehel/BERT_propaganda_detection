
import argparse

parser = argparse.ArgumentParser(description='PyTorch BERT model Training')

"----------------------------- General options -----------------------------"
parser.add_argument('--expID', default='default', type=str,
                    help='Experiment ID')
parser.add_argument('--trainDataset', default='', type=str,
                    help='CSV file containing the training dataset')
parser.add_argument('--evalDataset', default='', type=str,
                    help='Directory containing validation articles')                    
parser.add_argument('--testDataset', default='', type=str,
                    help='Directory containing test articles')                    
parser.add_argument('--snapshot', default=1, type=int,
                    help='How often to take a snapshot of the model (0 = never)')
parser.add_argument('--classType', default="single_label", type=str,
                    help='all_class | single_label | binary')
parser.add_argument('--binaryLabel', default=None, type=str,
                    help='One of the 18 classes')
parser.add_argument('--outputFile', default="pred_local.csv", type=str,
                    help='Directory and name of the file to store predictions.')
parser.add_argument('--bio', default=False, type=bool,
                    help='Activate bio encoding')
parser.add_argument('--bology', default=False, type=bool,
                    help='Run Bertology methods')
parser.add_argument('--techniques', default="tools/data/propaganda-techniques-names.txt", type=str,
                    help='Directory and name of the file that contains names of the techniques.')
parser.add_argument('--train', default = False, type=bool,
                    help='If set to False, only test on the given model')
parser.add_argument('--seed', default = 984, type=int,
                    help='Seed value')
parser.add_argument('--fp16', default = False, type=bool,
                    help='Half precision training')
"----------------------------- Model options -----------------------------"
parser.add_argument('--model', default=0, type=str,
                    help='Select a model to be trained: bert-base-cased|bert-base-uncased|bert-large-uncased|bert-large-cased')
parser.add_argument('--lowerCase', default=False, type=bool,
                    help='Set to true if using a uncased model')
parser.add_argument('--nLabels', default=21, type=int,
                    help='Number of labels to predict')
parser.add_argument('--loadModel', default=None, type=str,
                    help='Provide full path to a previously trained model')

parser.add_argument('--maxLen', default=256, type=float,
                    help='Max length of tokens in a single training sample')

"----------------------------- Hyperparameter options -----------------------------"
parser.add_argument('--LR', default=1e-3, type=float,
                    help='Learning rate')
parser.add_argument('--patience', default=7, type=float,
                    help='Patience count for early stopping')
parser.add_argument('--momentum', default=0, type=float,
                    help='Momentum')
parser.add_argument('--weightDecay', default=0, type=float,
                    help='Weight decay')



"----------------------------- Training options -----------------------------"
parser.add_argument('--nEpochs', default=10, type=int,
                    help='Number of epochs to train for')

parser.add_argument('--trainBatch', default=32, type=int,
                    help='Train-batch size')
parser.add_argument('--validBatch', default=32, type=int,
                    help='Valid-batch size')


opt = parser.parse_args()
