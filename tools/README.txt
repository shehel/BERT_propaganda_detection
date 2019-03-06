=======================

(Author: Preslav Nakov)

The tools v2 include the following:

- baseline systems

- Jupyter notebooks for the baseline systems, which show how to read and write the files

- command-line scoring scripts

- example how to call thes scoring scripts from Python:

- description of the evaluation measures

- a way to work with a custom local split of the training data

- visuzalizer for the spans for task 3

- the script that was used to generate the labels for task 2 from those for task 3


=======================


Below we describe how you can generate a custom split of the TRAIN data into
	TRAIN-train and TRAIN-dev

This would allow you to experiment locally, without a need to submit.
We further release several scripts that can help you get started or score your your system.

We release a baseline system for each task,
which shows how to read the input and how to produce an output in the correct format:
	example-submission-task1.py
	example-submission-task2.py
	example-submission-task3.py

We also release scorers for each task, which can be run locally:
	task1_scorer_onefile.py
	task2_scorer_onefile.py
	task3_scorer_onefile.py

Finally, we include a PDF document describing how the evaluation measures are computed:
	Evaluation_of_Propaganda_Techniques.pdf


1. You need to download the training data from 
	https://s3.us-east-2.amazonaws.com/propaganda-datathon/dataset/datasets-v4.zip
and than you have to unzip it in datasets-v4


2. You can generate the split:

	$ bash split-train.sh 


3. You can generate baseline predictions for each task and you can score them locally

	$ bash run-baseline-predictions-and-scores.sh


And there is a bonus: 


4. We have an example how to call the scorer from Python:

	example-calling-scorer-from-python.py


5. We have Jupyter notebooks for the baselines for the three tasks:

	example-submission-task1.ipynb
	example-submission-task2.ipynb
	example-submission-task3.ipynb


6. We also have code to visualize the spans for task 3.
	visualize_spans_for_task3.py


Here is how you can use it:
	$ python3 visualize_spans_for_task3.py \
		train-split/tasks-2-3/train-train/article711566593.task3.labels \
		train-split/tasks-2-3/train-train/article711566593.txt


7. Finally, in case you wonder how the labels for Task 2 
   were generated from those for Task 3, here is the script:
	
	make_gold_labels_for_task2.py


Good luck with the datathon!
