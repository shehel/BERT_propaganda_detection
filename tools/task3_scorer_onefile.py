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
import os.path
import re
import glob
import logging.handlers

TASK_3_ARTICLE_ID_COL=0
TASK_3_TECHNIQUE_NAME_COL=1
TASK_3_FRAGMENT_START_COL=2
TASK_3_FRAGMENT_END_COL=3
TECHNIQUE_NAMES_FILE=os.path.join("data","propaganda-techniques-names.txt")

logger = logging.getLogger("propaganda_scorer")
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(ch)


def check_data_file_lists(submission_annotations, gold_annotations, task_name="task3"):

    #checking that the number of articles for which the user has submitted annotations is correct
    if len(gold_annotations.keys()) < len(submission_annotations.keys()):
        logger.error("The number of articles in the submission, %d, is greater than the number of articles in the "
                     "reference dataset" % (len(submission_annotations.keys()), len(gold_annotations.keys()))); sys.exit()
    # logger.debug("OK: number of articles in the submission for %s is the same as the one in the gold file: %d"
    #              % (task_name, len(gold_annotations.keys())))

    #check that all file names are correct
    errors = [ article_id for article_id in submission_annotations.keys() if article_id not in gold_annotations.keys() ]
    if len(errors)>0:
        logger.error("The following article_ids in the submission file do not have a correspondence in the reference "
                     "dataset: %s\n" % (str(errors)))
    # gold_file_article_id_set = set([article_id for article_id in gold_annotations.keys()])
    # submission_file_article_id_set = set([article_id for article_id in submission_annotations.keys()])
    # intersection_file_list = gold_file_article_id_set.intersection(submission_file_article_id_set)
    # if len(intersection_file_list) != len(gold_annotations):
    #     logger.error("The list of article ids is not identical.\n"
    #              "The following article_ids in the submission file do not have a correspondence in the gold file: %s\n"
    #              "The following article_ids in the gold file do not have a correspondence in the submission file: %s"
    #              %(str(submission_file_article_id_set.difference(gold_file_article_id_set)),
    #                str(gold_file_article_id_set.difference(submission_file_article_id_set)))); sys.exit()
    logger.debug("OK: all article ids have a correspondence in the list of articles from the reference dataset")


def load_technique_names_from_file(filename=TECHNIQUE_NAMES_FILE):

    with open(filename, "r") as f:
        return [ line.rstrip() for line in f.readlines() ]


def extract_article_id_from_file_name(fullpathfilename):

    regex = re.compile("article([0-9]+).*")
    return regex.match(os.path.basename(fullpathfilename)).group(1)


def check_data_file_task3(submission_article_annotations, article_id, techniques_names):

    annotations = {}
    for i, row in enumerate(submission_article_annotations):
        if len(row) != 3:
            logger.error("Row %d in file %s is supposed to have 3 TAB-separated columns. Found %d."
                             % (i + 1, submission_article_annotations, len(row))); sys.exit()
        #checking the technique names are correct
        if row[0] not in techniques_names:
            logger.error("On row %d in file %s the technique name, %s, is incorrect. Possible values are: %s"
                             % (i + 1, submission_article_annotations, row[0], str(techniques_names))); sys.exit()
        #checking spans
        if int(row[1]) < 0 or int(row[2]) < 0:
            logger.error("On row %d in file %s, start and end of position of the fragment must be non-negative. "
                         "Found values %s, %s"
                             % (i + 1, submission_article_annotations, row[1], row[2])); sys.exit()
        if int(row[1]) >= int(row[2]):
            logger.error("On row %d in file %s, end position of the fragment must be greater than the starting "
                         "one. Found values %s, %s"%(i + 1, submission_article_annotations,
                                                     row[1], row[2])); sys.exit()
        #checking that there are no overlapping spans flagged with the same technique name
        if row[0] not in annotations.keys():
            annotations[row[0]] = []
        else:
            curr_span = set(range(int(row[1]), int(row[2])))
            for span in annotations[row[0]]:
                if len(set(range(int(span[0]), int(span[1]))).intersection(curr_span)) > 0:
                    logger.error("On row %d in article %s, the span of the annotation %s, [%s,%s] overlap with the "
                                 "following one from the same file: [%s,%s]"
                                 % (i + 1, article_id, row[0], row[1], row[2], span[0], span[1]));
                    sys.exit()
        annotations[row[0]].append([row[1], row[2]])
    logger.debug("OK: article %s format is correct" % (article_id))


def read_task3_output_file(filename):

    with open(filename, "r") as f:
        return [ line.rstrip().split("\t") for line in f.readlines() ]


def compute_technique_frequency(annotations_list, technique_name):
    return sum([ len([ example_annotation for example_annotation in x if example_annotation[0]==technique_name])
                 for x in annotations_list ])


def compute_score(submission_annotations, gold_annotations):

    prec_denominator = sum([len(annotations) for annotations in submission_annotations.values()])
    rec_denominator = sum([len(annotations) for annotations in gold_annotations.values()])
    technique_Spr = {propaganda_technique: 0 for propaganda_technique in load_technique_names_from_file()}
    cumulative_Spr = 0
    for article_id in submission_annotations.keys():
        gold_data = gold_annotations[article_id]
        logger.debug("Computing contribution to the score of article id %s\nand tuples %s\n%s\n"
                     % (article_id, str(submission_annotations[article_id]), str(gold_data)))
        for j, sd in enumerate(submission_annotations[article_id]): #submission_data:
            s=""
            sd_span = set(range(int(sd[1]), int(sd[2])))
            sd_annotation_length = int(sd[2]) - int(sd[1])
            for i, gd in enumerate(gold_data):
                if gd[0]==sd[0]:
                    #s += "\tmatch %s %s-%s - %s %s-%s"%(sd[0],sd[1], sd[2], gd[0], gd[1], gd[2])
                    gd_span = set(range(int(gd[1]), int(gd[2])))
                    intersection = len(sd_span.intersection(gd_span))
                    gd_annotation_length = int(gd[2]) - int(gd[1])
                    Spr = intersection/max(sd_annotation_length, gd_annotation_length)
                    cumulative_Spr += Spr
                    s += "\tmatch %s %s-%s - %s %s-%s: S(p,r)=|intersect(r, p)|/max(|p|,|r|) = %d/max(%d,%d) = %f (cumulative S(p,r)=%f)\n"\
                         %(sd[0],sd[1], sd[2], gd[0], gd[1], gd[2], intersection, sd_annotation_length, gd_annotation_length, Spr, cumulative_Spr)
                    technique_Spr[gd[0]] += Spr
            logger.debug("\n%s"%(s))

    p,r,f1=(0,0,0)
    if prec_denominator>0:
        p = cumulative_Spr/prec_denominator
    if rec_denominator>0:
        r = cumulative_Spr/rec_denominator
    logger.info("Precision=%f/%d=%f\tRecall=%f/%d=%f"
                 %(cumulative_Spr, prec_denominator, p, cumulative_Spr, rec_denominator, r))
    if prec_denominator == 0 and rec_denominator == 0:
        f1 = 1.0
    if p>0 and r>0:
        f1 = 2*(p*r/(p+r))
    logger.info("F1=%f"%(f1))

    for technique_name in technique_Spr.keys():
        prec_tech, rec_tech, f1_tech = (0,0,0)
        prec_tech_denominator = compute_technique_frequency(submission_annotations.values(), technique_name)
        rec_tech_denominator = compute_technique_frequency(gold_annotations.values(), technique_name)
        if prec_tech_denominator == 0 and rec_tech_denominator == 0: #
            f1_tech = 1.0
        else:
            if prec_tech_denominator > 0:
                prec_tech = technique_Spr[technique_name] / prec_tech_denominator
            if rec_tech_denominator > 0:
                rec_tech = technique_Spr[technique_name] / rec_tech_denominator
            if prec_tech>0 and rec_tech>0:
                f1_tech = 2*(prec_tech*rec_tech/(prec_tech+rec_tech))
        logger.info("F1-%s=%f"%(technique_name, f1_tech))

    return f1


def load_annotation_list_from_folder(folder_name):

    file_list = glob.glob(os.path.join(folder_name, "*.task3.labels"))
    if len(file_list)==0:
        logger.error("Cannot load file list in folder " + folder_name);
        sys.exit()
    annotations = {}
    for filename in file_list:
        annotations[extract_article_id_from_file_name(filename)] = []
        with open(filename, "r") as f:
            for line in f.readlines():
                row = line.rstrip().split("\t")
                annotations[row[TASK_3_ARTICLE_ID_COL]].append([ row[TASK_3_TECHNIQUE_NAME_COL],
                                                                 row[TASK_3_FRAGMENT_START_COL],
                                                                 row[TASK_3_FRAGMENT_END_COL] ])
    return annotations


def load_annotation_list_from_file(filename):

    annotations = {}
    with open(filename, "r") as f:
        for line in f.readlines():
            row = line.rstrip().split("\t")
            if row[TASK_3_ARTICLE_ID_COL] not in annotations.keys():
                annotations[row[TASK_3_ARTICLE_ID_COL]] = []
            annotations[row[TASK_3_ARTICLE_ID_COL]].append([ row[TASK_3_TECHNIQUE_NAME_COL],
                                                             row[TASK_3_FRAGMENT_START_COL],
                                                             row[TASK_3_FRAGMENT_END_COL]])
    return annotations


def main(args):

    user_submission_file = args.submission
    gold_folder = args.gold
    output_log_file = args.log_file

    if args.debug_on_std:
        ch.setLevel(logging.DEBUG)

    if output_log_file is not None:
        logger.info("Logging execution to file " + output_log_file)
        fileLogger = logging.FileHandler(output_log_file)
        fileLogger.setLevel(logging.DEBUG)
        fileLogger.setFormatter(formatter)
        logger.addHandler(fileLogger)

    submission_annotations = load_annotation_list_from_file(user_submission_file)
    techniques_names = load_technique_names_from_file()  # load technique names
    if gold_folder is None:
        # no gold file provided, perform only some checks on the submission files
        logger.info('Checking format of user submitted file %s' % (user_submission_file))
        for article_id in submission_annotations.keys():
            check_data_file_task3(submission_annotations[article_id], article_id, techniques_names)
        logger.warning("The format of the submitted file is ok. However, more checks, requiring the gold file, are needed "
                        "for the submission to be correct: the number of article and their ids must correspond to the "
                        "ones of the gold file, etc")
    else:
        logger.info('Checking user submitted file %s against gold folder %s' % (user_submission_file, gold_folder))
        gold_annotations = load_annotation_list_from_folder(gold_folder)
        check_data_file_lists(submission_annotations, gold_annotations)
        for article_id in submission_annotations.keys():
            check_data_file_task3(submission_annotations[article_id], article_id, techniques_names)
        logger.info('Scoring user submitted file %s against gold file %s' % (user_submission_file, gold_folder))
        return compute_score(submission_annotations, gold_annotations)
        #compute_score({user_file: read_task3_output_file(user_file) for user_file in submission_annotations},
        #              {gold_folder: read_task3_output_file(gold_folder) for gold_folder in gold_annotations})


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Scorer for 2019 Hackathon task 3. If only -s option is provided, only some checks on the "
                                     "submitted file is performed.")
    parser.add_argument('-s', '--submission-file', dest='submission', required=True, help="file with the submission of the team")
    parser.add_argument('-r', '--reference-folder', dest='gold', required=False, help="folder with the gold labels.")
    parser.add_argument('-d', '--enable-debug-on-standard-output', dest='debug_on_std', required=False,
                        action='store_true', help="Print debug info also on standard output.")
    parser.add_argument('-l', '--log-file', dest='log_file', required=False, help="Output logger file.")
    main(parser.parse_args())
