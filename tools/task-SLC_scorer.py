__author__ = "Giovanni Da San Martino"
__copyright__ = "Copyright 2019"
__credits__ = [""]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = ""
__email__ = "gmartino@hbku.edu.qa"
__status__ = "Beta"

import sys
import argparse
import logging.handlers
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

TASK_2_ARTICLE_ID_COL=0
TASK_2_SENTENCE_ID_COL=1
TASK_2_LABEL_COL=2
TASK_2_POSITIVE_LABEL = "propaganda"
TASK_2_NEGATIVE_LABEL = "non-propaganda"

logger = logging.getLogger("propaganda_scorer")
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.setLevel(logging.DEBUG)


def check_data_file_task2(submission_annotations, gold_annotations):

    #checking that the number of articles for which the user has submitted annotations is correct
    if len(gold_annotations.keys()) != len(submission_annotations.keys()):
        logger.debug("OK: number of articles in the submission for task 2 is the same as the one in the gold file: %d"
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
    logger.debug("OK: all article ids have a correspondence in the list of articles from the reference dataset")

    i, predictions, gold_labels = (0, [], [])
    for article_id in submission_annotations.keys():
        #checking that the number of sentences is the same
        if len(submission_annotations[article_id]) != len(gold_annotations[article_id]):
            logger.error("The number of sentences for article %s in the submission file and the gold file differ: %d - "
                         "%d" % (article_id, len(submission_annotations[article_id]), len(gold_annotations[article_id])))
            sys.exit()

        for submission_sentence, gold_sentence in zip(sorted(submission_annotations[article_id]),
                                                      sorted(gold_annotations[article_id])):
            i += 1
            #checking that sentence ids match
            if submission_sentence[0] != gold_sentence[0]:
                logger.error("On row %d the sentence id of the submission, %s, does not correspond to the one of the "
                             "gold file, %s"%(i, submission_sentence[0], gold_sentence[0]))
                sys.exit()
            #checking that the label is valid
            if submission_sentence[1].lower() not in [ TASK_2_POSITIVE_LABEL, TASK_2_NEGATIVE_LABEL ]:
                logger.error("On row %d the label of the submission, %s, is not valid. Possible values are: "
                             %(i, submission_sentence[1], [ TASK_2_POSITIVE_LABEL, TASK_2_NEGATIVE_LABEL ]))
            predictions.append(submission_sentence[1].lower())
            gold_labels.append(gold_sentence[1].lower())

    return predictions, gold_labels


def compute_score(submission_labels, gold_labels):

    precision = precision_score(gold_labels, submission_labels, pos_label=TASK_2_POSITIVE_LABEL)
    recall = recall_score(gold_labels, submission_labels, pos_label=TASK_2_POSITIVE_LABEL)
    f1 = f1_score(gold_labels, submission_labels, pos_label=TASK_2_POSITIVE_LABEL)

    return precision, recall, f1


def load_sentence_labels_from_file(filename):

    annotations = {}
    is_template_submission = False
    with open(filename, "r") as f:
        for i, line in enumerate(f.readlines()):
            row = line.rstrip().split("\t")
            if row[TASK_2_ARTICLE_ID_COL] not in annotations.keys():
                annotations[row[TASK_2_ARTICLE_ID_COL]] = []
            if len(row) != 3:
                logger.error("Row %d in file %s is supposed to have 3 TAB-separated columns. Found %d."
                             % (i + 1, filename, len(row)));
                sys.exit()
            if row[TASK_2_LABEL_COL]=="?":
                print(row[TASK_2_LABEL_COL])
                is_template_submission = True
            annotations[row[TASK_2_ARTICLE_ID_COL]].append([ row[TASK_2_SENTENCE_ID_COL], row[TASK_2_LABEL_COL] ])
    return annotations, is_template_submission


def main(args):

    user_submission_file = args.submission
    gold_file = args.gold
    output_log_file = args.log_file
    per_article_evaluation = bool(args.per_article_evaluation)
    output_for_script = bool(args.output_for_script)

    if not output_for_script:
        logger.addHandler(ch)

    if args.debug_on_std:
        ch.setLevel(logging.DEBUG)

    if output_log_file is not None:
        logger.info("Logging execution to file " + output_log_file)
        fileLogger = logging.FileHandler(output_log_file)
        fileLogger.setLevel(logging.DEBUG)
        fileLogger.setFormatter(formatter)
        logger.addHandler(fileLogger)

    submission_annotations, is_template_submission = load_sentence_labels_from_file(user_submission_file)
    gold_annotations, is_template_submission = load_sentence_labels_from_file(gold_file)
    #logger.info('Checking user submitted file %s against gold file %s' % (user_submission_file, gold_file))
    logger.info('Checking user submitted file %s' % (user_submission_file))
    predictions, gold_labels = check_data_file_task2(submission_annotations, gold_annotations)
    logger.info("OK: submission file %s format is correct" % (user_submission_file))
    if not is_template_submission:
        #logger.info('Scoring user submitted file %s against gold file %s' % (user_submission_file, gold_file))
        logger.info('Scoring user submitted file %s' % (user_submission_file))
        if per_article_evaluation:        
            f1_articles = []
            for article_id in sorted(gold_annotations.keys()):
                precision, recall, f1 = compute_score([ label for sent_id,label in submission_annotations[article_id] ], 
                                                      [ label for sent_id,label in gold_annotations[article_id] ])
                f1_articles.append(f1)
            print("per article evaluation F1=%s"%(",".join([ str(f1_value) for f1_value in  f1_articles])))
        else:
            precision, recall, f1 = compute_score(predictions, gold_labels)
            logger.info("\nPrecision=%f\nRecall=%f\nF1=%f" % (precision, recall, f1))
            if output_for_script:
                print("%f\t%f\t%f"%(f1, precision, recall))
    else:
        logger.info("Not scoring the submission because the gold file has ? instead of gold labels")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Scorer for SLC task on the Propaganda Techniques Corpus. ")
    parser.add_argument('-s', '--submission-file', dest='submission', required=True, help="file with the submission of the team")
    parser.add_argument('-r', '--reference-file', dest='gold', required=True, help="file with the gold labels.")
    parser.add_argument('-d', '--enable-debug-on-standard-output', dest='debug_on_std', required=False,
                        action='store_true', help="Print debug info also on standard output.")
    parser.add_argument('-l', '--log-file', dest='log_file', required=False, help="Output logger file.")
    parser.add_argument('-e', '--per-example-evaluation', dest='per_article_evaluation', required=False, action='store_true',
                        default=False, help="Prints the value of the evaluation function for each example/article")
    parser.add_argument('-o', '--output-for-script', dest='output_for_script', required=False, action='store_true',
                        default=False, help="Prints the output in an easy-to-parse way for a script")
    main(parser.parse_args())
