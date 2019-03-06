#!/bin/bash

## Author: Preslav Nakov

# We assume that you have downloaded the input data 
# from https://s3.us-east-2.amazonaws.com/propaganda-datathon/dataset/datasets-v4.zip
# and that you have unzipped it in datasets-v4 (you can also edit $DATA_DIR)

export DATA_DIR=datasets-v4

# The data split will be produced in TRAIN_SPLIT_DIR
export TRAIN_SPLIT_DIR=train-split
mkdir -p $TRAIN_SPLIT_DIR/task-1 || exit 1


## Task 1
echo Preparing the data for Task 1
sort $DATA_DIR/task-1/task1.train.txt | head -n 31993 \
	> $TRAIN_SPLIT_DIR/task-1/task1.train-train.txt || exit 2
sort $DATA_DIR/task-1/task1.train.txt | tail -n 4000 \
	> $TRAIN_SPLIT_DIR/task-1/task1.train-dev.txt || exit 3
cut -f 2,3 $TRAIN_SPLIT_DIR/task-1/task1.train-dev.txt \
	> $TRAIN_SPLIT_DIR/task-1/train-dev.task1.labels || exit 4
cut -f 2,3 $TRAIN_SPLIT_DIR/task-1/task1.train-train.txt \
	> $TRAIN_SPLIT_DIR/task-1/train-train.task1.labels || exit 5

## Tasks 2-3
echo Preparing the data for Tasks 2 and 3
mkdir -p $TRAIN_SPLIT_DIR/tasks-2-3 || exit 6
cp -r $DATA_DIR/tasks-2-3/train $TRAIN_SPLIT_DIR/tasks-2-3/train-train || exit 7
mkdir $TRAIN_SPLIT_DIR/tasks-2-3/train-dev || exit 8
cd $TRAIN_SPLIT_DIR/tasks-2-3/train-train || exit 9
for i in `ls -1 *.txt | head -n 43 | cut -d "." -f 1`; do mv $i.txt $i.task3.labels $i.task2.labels ../train-dev; done || exit 10
cd ../train-dev || exit 11
cat *.task2.labels > ../train-dev.task2.labels || exit 12
cat *.task3.labels > ../train-dev.task3.labels || exit 13
cd ../../.. || exit 14
cp $DATA_DIR/tasks-2-3/propaganda-techniques-names.txt $TRAIN_SPLIT_DIR/tasks-2-3/propaganda-techniques-names.txt || exit 15
