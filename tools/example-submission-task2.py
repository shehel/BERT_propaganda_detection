train_folder = "train-split/tasks-2-3/train-train" # if train and dev folders are not in the same folder as this notebook, change
dev_folder = "train-split/tasks-2-3/train-dev"     # these variables accordingly

from sklearn.linear_model import LogisticRegression
import glob
import os.path
import numpy as np

# loading articles' content from *.txt files in the train folder
file_list = glob.glob(os.path.join(train_folder, "*.txt"))
sentence_list = []
for i, filename in enumerate(file_list):
    with open(filename, "r", encoding="latin-1") as f:
        for row in f.readlines():
            sentence_list.append(row.rstrip())

# loading articles ids and sentence ids from files *.task2.labels in the train folder 
gold_file_list = glob.glob(os.path.join(train_folder, "*.task2.labels"))
articles_id, sentence_id_list, gold_labels = ([], [], [])
for filename in gold_file_list:
    with open(filename, "r") as f:
        for row in f.readlines():
            article_id, sentence_id, gold_label = row.rstrip().split("\t")
            articles_id.append(article_id)
            sentence_id_list.append(sentence_id)
            gold_labels.append(gold_label)
print("Loaded %d sentences from %d articles" % (len(sentence_list), i+1))

train = np.array([ len(sentence) for sentence in sentence_list ]).reshape(-1, 1)
model = LogisticRegression(penalty='l2', class_weight='balanced', solver="lbfgs")
model.fit(train, gold_labels)

file_list = glob.glob(os.path.join(dev_folder, "*.txt"))
dev_sentence_list = []
for i, filename in enumerate(file_list):
    with open(filename, "r", encoding="latin-1") as f:
        for row in f.readlines():
            dev_sentence_list.append(len(row.rstrip()))

gold_file_list = glob.glob(os.path.join(dev_folder, "*.task2.labels"))
dev_articles_id, dev_sentence_id_list = ([], [])
for filename in gold_file_list:
    with open(filename, "r") as f:
        for row in f.readlines():
            article_id, sentence_id = row.rstrip().split("\t")[0:2]
            dev_articles_id.append(article_id)
            dev_sentence_id_list.append(sentence_id)

dev = np.array(dev_sentence_list).reshape(-1, 1)
predictions = model.predict(dev)

with open("example-submission-task2-predictions.txt", "w") as fout:
    for article_id, sentence_id, prediction in zip(dev_articles_id, dev_sentence_id_list, predictions):
        fout.write("%s\t%s\t%s\n" % (article_id, sentence_id, prediction))
