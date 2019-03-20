# task3
Sequence classification for propaganda dataset (QCRI)

1. pip install -r requirements.txt 
2. Run python -m spacy download en
3. dataset_train.csv and dataset_dev.csv are created from datasets_v5 by running preprocess.py on it. To create a different 
dataset, run preprocess.py to generate train and dev files.
eg: <br>
```python preprocess.py -d [path to articles and labels directory] -o [name of output file] -s Propaganda```
<br>-s and -b are additional flags used to get binary label dataset and bio formatted dataset respectively. 
4. Create folder ```./exp``` - This is where the logs and model states will be stored for training runs. 
5. Run the trainer, for example <br>
```python train.py --expID test_run --trainDataset dataset_train.csv --valDataset dataset_dev.csv --model bert-base-cased --LR 3e-5 --trainBatch 12 --nEpochs 5 --classType binary --nLabels 4```

6. Run python predict.py to get output in the character level. For example: <br>
```python predict.py --valDataset datasets-v5/tasks-2-3/dev/ --model bert-base-cased --validBatch 12 --loadModel exp/binary/binary_2E/1/model_1.pth --outputFile pred.csv --classType binary --nLabels 4```

## Tested on:
huggingface/pytorch-pretrained-BERT 0.4 <br>
Pandas 0.24.1 <br>
Spacy 2.0.18 <br>
Torch 1.0 <br>
<br>
Python 3.6.8 <br>
CUDA 9

# Evaluation 

Default Task: identification of fragments and techniques
1. ```cd tools```
2. Assuming the predictions are in file dev.labels and the gold labels in the folder task3-gold-labels/dev-task3-labels, the following command evaluates the predictions on the development set 
```python task3_scorer_onefile.py -s dev.labels -r task3-gold-labels/dev-task3-labels -t propaganda-techniques-names.txt```

Fragment identification only task (two fragments are considered to match no matter what their associated technique is)
1. ```cd tools```
2. Assuming the predictions are in file dev.labels and the gold labels in the folder task3-gold-labels/dev-task3-labels, the following command evaluates the predictions on the development set 
```python task3_scorer_onefile.py -s dev.labels -r task3-gold-labels/dev-task3-labels -t propaganda-techniques-names.txt -f```
Notice that if the file with predictions has overlapping spans, an error is raised and no scoring is computed 
