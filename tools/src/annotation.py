
__author__ = "Giovanni Da San Martino"
__copyright__ = "Copyright 2019"
__credits__ = ["Giovanni Da San Martino"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Giovanni Da San Martino"
__email__ = "gmartino@hbku.edu.qa"
__status__ = "Beta"


import sys
import src.propaganda_techniques as pt
import logging.handlers

logger = logging.getLogger("propaganda_scorer")


class Annotation():

    # input file format variables
    separator = "\t"
    ARTICLE_ID_COL = 0
    TECHNIQUE_NAME_COL = 1
    FRAGMENT_START_COL = 2
    FRAGMENT_END_COL = 3
    propaganda_techniques = None


    def __init__(self, label=None, start_offset = None, end_offset=None): #, article_id=None):
        
        self.label = label
        self.start_offset = int(start_offset)
        self.end_offset = int(end_offset)
        #self.article_id = article_id


    def __str__(self):

        return "%s\t%d\t%d"%(self.get_label(), self.start_offset, self.end_offset)


    def get_label(self):

        return self.label


    def get_start_offset(self):

        return self.start_offset

        
    def get_end_offset(self):

        return self.end_offset

    
    def get_span(self):
        """

        :return: a set of positions of all characters
        """
        return set(range(self.get_start_offset(), self.get_end_offset()))

    @staticmethod
    def load_annotation_from_string(annotation_string, row_num=None, filename=None):
        """
        Read annotations from a string in the following fields, separated
        by the class variable separator:

        article id, technique name, starting_position<separator>ending_position
        Fields order is determined by the class variables ARTICLE_ID_COL,
        TECHNIQUE_NAME_COL, FRAGMENT_START_COL, FRAGMENT_END_COL

        :return (an Annotation object, the id of the article)
        """

        row = annotation_string.rstrip().split(Annotation.separator)
        if len(row) != 4:
            logger.error("Row%s%s is supposed to have 4 columns. Found %d: -%s-."
                         % (" " + str(row_num) if row_num is not None else "",
                            " in file " + filename if filename is not None else "", len(row), annotation_string))
            sys.exit()

        article_id = row[Annotation.ARTICLE_ID_COL]
        label = row[Annotation.TECHNIQUE_NAME_COL]
        try:
            start_offset = int(row[Annotation.FRAGMENT_START_COL])
        except:
            logger.error("The column %d in row%s%s is supposed to be an integer: -%s-"
                         %(Annotation.FRAGMENT_START_COL, " " + str(row_num) if row_num is not None else "",
                            " in file " + filename if filename is not None else "", annotation_string))
        try:
            end_offset = int(row[Annotation.FRAGMENT_END_COL])
        except:
            logger.error("The column %d in row%s%s is supposed to be an integer: -%s-"
                         %(Annotation.FRAGMENT_END_COL, " " + str(row_num) if row_num is not None else "",
                            " in file " + filename if filename is not None else "", annotation_string))

        return Annotation(label, start_offset, end_offset), article_id

    def merge_spans(self, annotations_without_overlapping, i):
        """

        :param annotations_without_overlapping: a list of annotations of an article
        :param i: the index in spans which needs to be tested for overlapping
        :return:
        """
        annotations_without_overlapping[j][1] = annotations_without_overlapping[j][1].union(
                    annotations_without_overlapping[i][1])
        del (annotations_without_overlapping[i])

    def span_overlapping(self, second_annotation):
        return len(self.get_span().intersection(second_annotation.get_span())) > 0

    def check_format_of_annotation_in_file(self):
        """
        Performs some checks on the fields of the annotation
        """
        # checking the technique names are correct
        if not self.propaganda_techniques.is_valid_technique(self.get_label()):
            logger.error("label %s is not valid. Possible values are: %s"%(self.get_label(), self.propaganda_techniques))
            sys.exit()
        # checking spans
        if self.get_start_offset() < 0 or self.get_end_offset() < 0:
            logger.error("Start and end of position of the fragment must be non-negative: %d, %d"
                         %(self.get_start_offset(), self.get_end_offset()))
            sys.exit()
        if self.get_start_offset() >= self.get_end_offset():
            logger.error("End position of the fragment must be greater than the starting one: start=%d, end=%d"
                         %(self.get_start_offset(), self.get_end_offset()))
            sys.exit()
