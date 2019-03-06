#!/bin/bash

## Author: Preslav Nakov

export TRAIN_SPLIT_DIR=train-split


# Testing Task 1
echo Generating baseline prediction for task 1
python3 example-submission-task1.py || exit 1
mv example-submission-task1-predictions.txt $TRAIN_SPLIT_DIR/task-1/ || exit 2

echo Testing task 1
python3 task1_scorer_onefile.py \
	-s $TRAIN_SPLIT_DIR/task-1/example-submission-task1-predictions.txt \
	-r $TRAIN_SPLIT_DIR/task-1/train-dev.task1.labels \
	-l $TRAIN_SPLIT_DIR/task-1/example-submission-task1-predictions.txt.log || exit 3



# Testing Task 2
echo Generating baseline prediction for task 2
python3 example-submission-task2.py || exit 4
mv example-submission-task2-predictions.txt $TRAIN_SPLIT_DIR/tasks-2-3/ || exit 5

echo Testing task 2
python3 task2_scorer_onefile.py \
	-s $TRAIN_SPLIT_DIR/tasks-2-3/example-submission-task2-predictions.txt \
	-r $TRAIN_SPLIT_DIR/tasks-2-3/train-dev.task2.labels \
	-l $TRAIN_SPLIT_DIR/tasks-2-3/example-submission-task2-predictions.txt.log || exit 6


## Testing Task 3
echo Generating baseline prediction for task 3
python3 example-submission-task3.py || exit 7
mv example-submission-task3-predictions.txt $TRAIN_SPLIT_DIR/tasks-2-3/ || exit 8

echo Testing task 3
python3 task3_scorer_onefile.py \
	-s $TRAIN_SPLIT_DIR/tasks-2-3/example-submission-task3-predictions.txt \
	-r $TRAIN_SPLIT_DIR/tasks-2-3/train-dev \
	-l $TRAIN_SPLIT_DIR/tasks-2-3/example-submission-task3-predictions.txt.log || exit 9
