import src.article_annotations as an
import src.annotation as ans
import glob
import os.path
import sys

import logging.handlers
logger = logging.getLogger("propaganda_scorer")


class Annotations:
    """
    Dictionary of article Annotation objects
    """

    def __init__(self, annotations=None):

        if annotations is None:
            self.annotations = {} # list of Annotation objects
        else:
            self.annotations = annotations


    def __len__(self):

        return len(self.annotations.keys())


    def add_annotation(self, annotation, article_id):

        if not self.has_article(article_id):
            self.annotations[article_id] = an.Articles_annotations(article_id=article_id)
        self.annotations[article_id].add_annotation(annotation)


    def check_annotation_spans_with_category_matching(self, merge_overlapping_spans=False):
        """
        Check whether there are overlapping spans for the same technique in the same article.
        Two spans are overlapping if their associated techniques match (according to category_matching_func)
        If merge_overlapping_spans==True then the overlapping spans are merged, otherwise an error is raised.

        :param merge_overlapping_spans: if True merges the overlapping spans
        :return:
        """

        for article_id in self.get_article_id_list():

            annotation_list = self.get_article_annotations(article_id).groupby_technique()
            if merge_overlapping_spans:
                for technique in annotation_list.keys():
                    for i in range(1, len(annotation_list[technique])):
                        annotation_list[technique][i].merge_spans(annotation_list[technique], i-1)
            if not self.get_article_annotations(article_id):
                return False
            # annotation_list = {}
            # for annotation in self.annotations.get_article_annotations(article_id):
            #     technique = annotation.get_label()
            #     if technique not in annotation_list.keys():
            #         annotation_list[technique] = [[technique, curr_span]]
            #     else:
            #         if merge_overlapping_spans:
            #             annotation_list[technique].append([technique, curr_span])
            #             merge_spans(annotation_list[technique], len(annotation_list[technique]) - 1)
            #         else:
            #             for matching_technique, span in annotation_list[technique]:
            #                 if len(curr_span.intersection(span)) > 0:
            #                     logger.error("In article %s, the span of the annotation %s, [%s,%s] overlap with "
            #                                  "the following one from the same article:%s, [%s,%s]" % (
            #                                  article_id, matching_technique,
            #                                  min(span), max(span), technique, min(curr_span), max(curr_span)))
            #                     return False
            #             annotation_list[technique].append([technique, curr_span])
            # if merge_overlapping_spans:
            #     annotations[article_id] = []
            #     for technique in annotation_list.keys():
            #         annotations[article_id] += annotation_list[technique]
        return True


    def has_article(self, article_id):

        return article_id in self.annotations.keys()


    def get_article_id_list(self):

        return self.annotations.keys()


    def get_article_annotations(self, article_id):

        return self.annotations[article_id]


    def load_annotation_list_from_file(self, filename):

        with open(filename, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                ann, article_id = ans.Annotation.load_annotation_from_string(line.rstrip(), i, filename)
                ann.check_format_of_annotation_in_file()
                self.add_annotation(ann, article_id)
        #return anns


    def load_annotation_list_from_folder(self, folder_name, pattern="*.labels"):

        file_list = glob.glob(os.path.join(folder_name, pattern))
        if len(file_list) == 0:
            logger.error("Cannot load file list %s/%s"%(folder_name, pattern))
            sys.exit()
        for filename in file_list:
            self.load_annotation_list_from_file(filename)
        #     annotations[extract_article_id_from_file_name(filename)] = []
        #     with open(filename, "r") as f:
        #         for row_number, line in enumerate(f.readlines()):
        #             row = line.rstrip().split("\t")
        #             check_format_of_annotation_in_file(row, row_number, techniques_names, filename)
        #             # annotations[row[TASK_3_ARTICLE_ID_COL]].append([ row[TASK_3_TECHNIQUE_NAME_COL],
        #             #                                                  row[TASK_3_FRAGMENT_START_COL],
        #             #                                                  row[TASK_3_FRAGMENT_END_COL] ])
        #             annotations[row[TASK_3_ARTICLE_ID_COL]].append([row[TASK_3_TECHNIQUE_NAME_COL],
        #                                                             set(range(int(row[TASK_3_FRAGMENT_START_COL]),
        #                                                                       int(row[TASK_3_FRAGMENT_END_COL])))])
        # return annotations


    def compute_technique_frequency(annotations_list, technique_name):
        return sum([len([example_annotation for example_annotation in x if example_annotation[0] == technique_name])
                    for x in annotations_list])


    def print_annotations(annotation_list):
        s = ""
        i=0
        for technique, span in annotation_list:
            s += "%d) %s: %d - %d\n"%(i, technique, min(span), max(span))
            i += 1
        return s
