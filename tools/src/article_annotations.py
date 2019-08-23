__author__ = "Giovanni Da San Martino"
__copyright__ = "Copyright 2019"
__credits__ = ["Giovanni Da San Martino"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Giovanni Da San Martino"
__email__ = "gmartino@hbku.edu.qa"
__status__ = "Beta"


import sys
import src.annotation as ans
import logging.handlers
logger = logging.getLogger("propaganda_scorer")


class Articles_annotations():

    start_annotation_effect = "\033[43;33m"
    end_annotation_effect = "\033[m"
    start_annotation_str = "{"
    end_annotation_str = "}"
    annotation_background_color = "\033[44;33m"


    def __init__(self, spans=None, article_id=None):

        if spans is None:
            self.spans = []
        else:
            self.spans = spans
        self.article_id = article_id


    def __len__(self):

        return len(self.spans)


    def __str__(self):

        return "article id: %s\n%s"%(self.article_id, "\n".join(self.spans))


    def add_annotation(self, annotation, article_id=None):

        self.add_article_id(article_id)
        self.spans.append(annotation)


    def add_article_id(self, article_id):

        if self.article_id is None:
            self.article_id = article_id
        else:
            if article_id is not None and self.article_id != article_id:
                logger.error("Trying to add an annotation with a different article id")
                sys.exit()

    def get_article_id(self):

        return self.article_id

    def get_article_annotations(self):

        return self.spans


    def get_markers_from_spans(self):

        self.sort_spans()
        self.markers = []
        for i, annotation in enumerate(self.spans, 1):
            self.markers.append((annotation.get_start_offset(), annotation.get_label(), i, "start"))
            self.markers.append((annotation.get_end_offset(), annotation.get_label(), i, "end"))
        self.markers = sorted(self.markers, key=lambda ann: ann[0])


    def groupby_technique(self):

        annotation_list = {}
        for i, annotation in enumerate(self.get_article_annotations()):
            technique = annotation.get_label()
            if technique not in annotation_list.keys():
                annotation_list[technique] = []
            annotation_list[technique].insert(0, i)
        return annotation_list


    #check_annotation_spans_with_category_matching
    def has_overlapping_spans(self, merge_overlapping_spans=False):
        """
        Check whether there are ovelapping spans for the same technique in the same article.
        Two spans are overlapping if their associated techniques match (according to category_matching_func)
        If merge_overlapping_spans==True then the overlapping spans are merged, otherwise an error is raised.

        :param merge_overlapping_spans: if True merges the overlapping spans
        :return:
        """

        annotation_list = {}
        for annotation in self.get_article_annotations():
            technique = annotation.get_label()
            if technique not in annotation_list.keys():
                annotation_list[technique] = [annotation] #[[technique, curr_span]]
            else:
                if merge_overlapping_spans:
                    annotation_list[technique].append(annotation)
                    self.merge_spans(annotation_list[technique], len(annotation_list[technique]) - 1)
                else:
                    for matching_technique, span in annotation_list[technique]:
                        if len(curr_span.intersection(span)) > 0:
                            logger.error("In article %s, the span of the annotation %s, [%s,%s] overlap with "
                                         "the following one from the same article:%s, [%s,%s]" % (
                                             article_id, matching_technique,
                                             min(span), max(span), technique, min(curr_span), max(curr_span)))
                            return False
                    annotation_list[technique].append([technique, curr_span])
        if merge_overlapping_spans:
            annotations[article_id] = []
            for technique in annotation_list.keys():
                annotations[article_id] += annotation_list[technique]
        return True


    def is_starting_marker(self, marker_index=None):

        if marker_index is None:
            marker_index = self.curr_marker
        if marker_index < len(self.markers):
            return self.markers[marker_index][3] == "start"


    def is_ending_marker(self, marker_index=None):

        if marker_index is None:
            marker_index = self.curr_marker
        if marker_index < len(self.markers):
            return self.markers[marker_index][3] == "end"


    def load_article_annotations_from_csv_file(self, filename):
        """
        Read annotations from a csv file and creates a list of
        Annotation objects. Check class annotation for details
        on the file format.
        """
        with open(filename, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                an, article_id = ans.Annotation.load_annotation_from_string(line.rstrip(), i, filename)
                self.add_annotation(an, article_id)


    def marker_label(self, marker_index=None):

        if marker_index is None:
            marker_index = self.curr_marker
        if marker_index < len(self.markers):
            return self.markers[marker_index][1]
        # else:
        # ERROR


    def marker_position(self, marker_index=None):

        if marker_index is None:
            marker_index = self.curr_marker
        if marker_index < len(self.markers):
            return self.markers[marker_index][0]


    def marker_annotation(self, marker_index=None):

        if marker_index is None:
            marker_index = self.curr_marker
        if marker_index < len(self.markers):
            return self.markers[marker_index][2]


    def mark_text(self, original_text, print_line_numbers=False):
        """
        mark the string original_text with object's annotations

        original_text: string with the text to be marked
        print_line_numbers: add line numbers to the text

        :return output_text the text in string original_text with added marks
                footnotes the list of techniques in the text
                legend description of the marks added
        """

        self.get_markers_from_spans()

        output_text, curr_output_text_index, self.curr_marker = ("", 0, 0)
        footnotes = "List of techniques found in the article\n\n"
        annotations_stack = []  # to handle overlapping annotations when assigning color background
        while curr_output_text_index < len(original_text):
            if self.curr_marker >= len(self.markers):
                output_text += original_text[curr_output_text_index:]
                curr_output_text_index = len(original_text)
            else:
                if self.marker_position() <= curr_output_text_index:
                    if self.is_starting_marker():
                        output_text += self.start_annotation_effect + self.start_annotation_str
                        annotations_stack.append(self.marker_annotation())
                        footnotes += "%d: %s\n" % (self.marker_annotation(), self.marker_label())
                    else:
                        output_text += "%s%s%s" % (
                            self.end_annotation_effect, "" if len(annotations_stack) > 1 else " ",
                            self.start_annotation_effect)
                    output_text += str(self.marker_annotation())
                    if self.is_ending_marker():
                        output_text += self.end_annotation_str + self.end_annotation_effect
                        annotations_stack.remove(self.marker_annotation())
                        if len(annotations_stack) > 0:
                            output_text += self.annotation_background_color
                    else:
                        output_text += self.end_annotation_effect + " " + self.annotation_background_color
                    self.curr_marker += 1
                else:
                    output_text += original_text[curr_output_text_index:self.marker_position()]
                    curr_output_text_index = self.marker_position()

        if print_line_numbers:
            output_text = "\n".join([str(i) + " " + line for i, line in enumerate(output_text.split("\n"), 1)])

        legend = "---\n%sHighlighted text%s: any propagandistic fragment\n%s%si%s: start of the i-th technique" \
                 "\n%si%s%s: end of the i-th technque\n---"\
                 %(self.annotation_background_color, self.end_annotation_effect, self.start_annotation_effect,
                   self.start_annotation_str, self.end_annotation_effect, self.start_annotation_effect,
                   self.end_annotation_str, self.end_annotation_effect)

        return output_text, footnotes, legend


    def merge_spans(self, annotations_without_overlapping, i):
        """
        Checks if annotations_without_overlapping
        :param annotations_without_overlapping: a list of Annotations objects of an article assumed to be
                without overlapping
        :param i: the index in spans which needs to be tested for overlapping
        :return:
        """
        #print("checking element %d of %d"%(i, len(spans)))
        if i<0:
            return True
        for j in range(0, i): #len(annotations_without_overlapping)):
            assert i<len(annotations_without_overlapping) or print(i, len(annotations_without_overlapping))
            if j != i and annotations_without_overlapping[i].span_overlapping(annotations_without_overlapping[j]):
                 #   len(annotations_without_overlapping[i][1].intersection(annotations_without_overlapping[j][1])) > 0:
                # print("Found overlapping spans: %d-%d and %d-%d in annotations %d,%d:\n%s"
                #       %(min(annotations_without_overlapping[i][1]), max(annotations_without_overlapping[i][1]),
                #         min(annotations_without_overlapping[j][1]), max(annotations_without_overlapping[j][1]), i,j,
                #         print_annotations(annotations_without_overlapping)))
                annotations_without_overlapping[j][1] = annotations_without_overlapping[j][1].union(annotations_without_overlapping[i][1])
                del(annotations_without_overlapping[i])
                # print("Annotations after deletion:\n%s"%(print_annotations(annotations_without_overlapping)))
                if j > i:
                    j -= 1
                # print("calling recursively")
                self.merge_spans(annotations_without_overlapping, j)
                # print("done")
                return True

        return False


    def remove_empty_annotations(self):

        self.spans = [ span for span in self.spans if span is not None ]


    def sort_spans(self):
        """
        sort a set of annotations read with
        """
        self.spans = sorted(self.spans, key=lambda span: span.get_start_offset() )


