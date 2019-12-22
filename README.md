# task3
Sequence classification for propaganda dataset (QCRI)


1. ```pip install -r requirements.txt```
2. ```python -m spacy download en```

## Train
3. To create train, dev sets out of the training data: ```sh tools/split-train.sh``` 
4. Raw dataset is converted into intermediate pickle files by running preprocess.py on it. Run preprocess.py to generate train and dev files.
eg: <br>
```python preprocess.py -d [path to articles and labels directory] -o [name of output file] -l```
<br>-l flag preserves labels if included (needed even when labels aren't available).  
5. Run the trainer, for example <br>

```python train.py --expID test_run1--trainDataset train-train.p --evalDataset train-dev.p --model bert --LR 3e-5 --trainBatch 32 --nEpochs 5 --classType all_class --nLabels 21 --testDataset datasets-v2/tasks-2-3/dev --train True --lowerCase True & ``` <br>
Here, train.p and dev.p is obtained by running ```preprocess.py```. <br>
6. ```./exp``` directory contains the logs and model states for training runs. 

## Evaluation and Testing
A trained model can be tested on a dataset using 
```python train.py --expID test_run1 --trainDataset train-train.p --evalDataset train-dev.p --model bert --LR 3e-5 --trainBatch 32 --nEpochs 5 --classType all_class --nLabels 21 --testDataset datasets-v2/tasks-2-3/dev --lowerCase True --loadModel exp/all_class/test_run1/ &```. Doing so will use the best model based on the F1 score on validation set during train. A model can be used to produce predictions on a test by first creating the binarized pickle file and then using the previous command. The output will be in the directory containing the model state labelled ```pred.[test dir name]```.

## Tested on:
QCRI dataset V2 (NLP4IF) 
huggingface/pytorch-pretrained-BERT **1.0**<br>
Pandas 0.25.3 <br>
Spacy 2.0.18 <br>
Torch 1.3.1 <br>
<br>
Python 3.7 <br>
CUDA 10.1


