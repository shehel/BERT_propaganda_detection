train_set_file_name = "train-split/task-1/task1.train-train.txt" # if train and dev sets are not in current folder, change 
dev_set_file_name = "train-split/task-1/task1.train-dev.txt"     # these variables accordingly

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))

articles_content, gold_labels = ([], [])
with open(train_set_file_name, "r") as f:
        for line in f.readlines():
            article_content, article_id, gold_label = line.rstrip().split("\t")
            articles_content.append(article_content)
            gold_labels.append(gold_label)
print("Number of documents in the training set: %d"%(len(articles_content)))

dev_articles_content, dev_articles_id = ([], [])
with open(dev_set_file_name) as f:
    for line in f.readlines():
            article_content, article_id, gold_label = line.rstrip().split("\t")
            dev_articles_content.append(article_content)
            dev_articles_id.append(article_id)

train = vectorizer.fit_transform(articles_content)
dev = vectorizer.transform(dev_articles_content)
print("Checking that the number of features in train and dev correspond: %s - %s" % (train.shape[1], dev.shape[1]))

model = LogisticRegression(penalty='l2', class_weight='balanced', solver="lbfgs")
model.fit(train, gold_labels)
predictions = model.predict(dev)

with open("example-submission-task1-predictions.txt", "w") as fout:
    for article_id, prediction in zip(dev_articles_id, predictions):
        fout.write("%s\t%s\n" % (article_id, prediction))

