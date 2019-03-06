# -*- coding: UTF-8 -*-

__author__ = "Giovanni Da San Martino"
__copyright__ = "Copyright 2019"
__credits__ = ["Giovanni Da San Martino"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Giovanni Da San Martino"
__email__ = "gmartino@qf"
__status__ = "Beta"

import sys
import argparse
import logging.handlers
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

TASK_1_ARTICLE_ID_COL=0
TASK_1_LABEL_COL=1
TASK_1_POSITIVE_LABEL = "propaganda"
TASK_1_NEGATIVE_LABEL = "non-propaganda"

logger = logging.getLogger("propaganda_scorer")
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(ch)


def check_data_file_task1(submission_annotations, gold_annotations):

    #checking that the number of articles for which the user has submitted annotations is correct
    if len(gold_annotations.keys()) != len(submission_annotations.keys()):
        logger.debug("OK: number of articles in the submission for task 1 is the same as the one in the gold file: %d"
                     % (len(gold_annotations.keys())))
    #check that all article ids are correct
    gold_file_article_id_set = set([article_id for article_id in gold_annotations.keys()])
    submission_file_article_id_set = set([article_id for article_id in submission_annotations.keys()])
    intersection_file_list = gold_file_article_id_set.intersection(submission_file_article_id_set)
    if len(intersection_file_list) != len(gold_annotations):
        logger.error("The list of article ids is not identical.\n"
                 "The following article_ids in the submission file do not have a correspondence in the gold file: %s\n"
                 "The following article_ids in the gold file do not have a correspondence in the submission file: %s"
                 %(str(submission_file_article_id_set.difference(gold_file_article_id_set)),
                   str(gold_file_article_id_set.difference(submission_file_article_id_set)))); sys.exit()
    logger.debug("OK: all article ids have a correspondence in the list of articles from the gold file")


def compute_score(submission_annotations, gold_annotations):

    submission_labels, gold_labels = ([], [])
    for article_id in submission_annotations.keys():
        submission_labels.append(submission_annotations[article_id])
        gold_labels.append(gold_annotations[article_id])

    precision = precision_score(gold_labels, submission_labels, pos_label=TASK_1_POSITIVE_LABEL)
    recall = recall_score(gold_labels, submission_labels, pos_label=TASK_1_POSITIVE_LABEL)
    f1 = f1_score(gold_labels, submission_labels, pos_label=TASK_1_POSITIVE_LABEL)

    return precision, recall, f1


def load_sentence_labels_from_file(filename):

    annotations = {}
    with open(filename, "r") as f:
        for i, line in enumerate(f.readlines()):
            row = line.rstrip().split("\t")
            if len(row) != 2:
                logger.error("Row %d in file %s is supposed to have 2 TAB-separated columns. Found %d."
                             % (i + 1, filename, len(row)));
                sys.exit()
            if row[TASK_1_ARTICLE_ID_COL] in annotations.keys():
                logger.error("In row %d of file %s found a duplicated article id: %s" % (i+1, filename, row[TASK_1_ARTICLE_ID_COL]))
                sys.exit()
            if row[TASK_1_LABEL_COL] not in [ TASK_1_POSITIVE_LABEL, TASK_1_NEGATIVE_LABEL ]:
                logger.error("In row %d of file %s the label %s is not valid. Possible values are %s"
                             % (i+1, filename, row[TASK_1_LABEL_COL], str([TASK_1_POSITIVE_LABEL, TASK_1_NEGATIVE_LABEL])))
                sys.exit()
            annotations[row[TASK_1_ARTICLE_ID_COL]] = row[TASK_1_LABEL_COL]
    return annotations


def main(args):

    user_submission_file = args.submission
    gold_file = args.gold
    output_log_file = args.log_file

    if args.debug_on_std:
        ch.setLevel(logging.DEBUG)

    if output_log_file is not None:
        logger.info("Logging execution to file " + output_log_file)
        fileLogger = logging.FileHandler(output_log_file)
        fileLogger.setLevel(logging.DEBUG)
        fileLogger.setFormatter(formatter)
        logger.addHandler(fileLogger)

    submission_annotations = load_sentence_labels_from_file(user_submission_file)
    gold_annotations = load_sentence_labels_from_file(gold_file)
    logger.info('Checking user submitted file %s against gold file %s' % (user_submission_file, gold_file))
    check_data_file_task1(submission_annotations, gold_annotations)
    logger.info("OK: submission file %s format is correct" % (user_submission_file))
    logger.info('Scoring user submitted file %s against gold file %s' % (user_submission_file, gold_file))
    precision, recall, f1 = compute_score(submission_annotations, gold_annotations)
    logger.info("\nPrecision=%f\nRecall=%f\nF1=%f" % (precision, recall, f1))


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Scorer for 2019 Hackathon task 2. ")
    parser.add_argument('-s', '--submission-file', dest='submission', required=True, help="file with the submission of the team")
    parser.add_argument('-r', '--reference-file', dest='gold', required=True, help="file with the gold labels.")
    parser.add_argument('-d', '--enable-debug-on-standard-output', dest='debug_on_std', required=False,
                        action='store_true', help="Print debug info also on standard output.")
    parser.add_argument('-l', '--log-file', dest='log_file', required=False, help="Output logger file.")
    main(parser.parse_args())
