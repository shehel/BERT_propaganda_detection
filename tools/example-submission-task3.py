dev_folder = "train-split/tasks-2-3/train-dev" # if dev folder and propaganda-techniques-names.txt are not in the same folder as this 
propaganda_techniques_file = "train-split/tasks-2-3/propaganda-techniques-names.txt" # notebook, change these variables accordingly

import glob
import os.path
import random
import re

regex = re.compile("article([0-9]+).*") # regular expression for extracting article id from file name
random.seed(10) # to make runs deterministic

# loading articles' content from *.txt files in the dev folder
file_list = glob.glob(os.path.join(dev_folder, "*.txt"))
articles_content, articles_id = ([], [])
for filename in file_list:
    with open(filename, "r", encoding="latin-1") as f:  # Although there are only slight differences with respect 
                                                        # to UTF-8 encoding, in order for the counting of the 
                                                        # spans to be exactly consistent with the annotations,
                                                        # the file must be opened with latin-1 encoding. 
        articles_content.append(f.read())
        articles_id.append(regex.match(os.path.basename(filename)).group(1)) # extract article id from file name

with open(propaganda_techniques_file, "r") as f:
    propaganda_techniques_names = [ line.rstrip() for line in f.readlines() ]

with open("example-submission-task3-predictions.txt", "w") as fout:
    for article_content, article_id in zip(articles_content, articles_id):
        start_fragment, end_fragment, article_length = (0, 0, len(article_content))
        current_article_annotations = []
        while end_fragment < article_length:
            if end_fragment > 0:
                technique_name = propaganda_techniques_names[random.randint(0, len(propaganda_techniques_names)-1)]
                # check that there is no other annotation for the same anrticle and technique that overlaps
                intersection_length = 0
                if len(current_article_annotations) > 0:
                    span_annotation = set(range(start_fragment, end_fragment))
                    intersection_length = sum( [ len(span_annotation.intersection(previous_fragment))
                             for previous_technique, previous_fragment in current_article_annotations 
                             if previous_technique==technique_name ])
                if len(current_article_annotations) == 0 or intersection_length > 0:
                    fout.write("%s\t%s\t%s\t%s\n" % (article_id, technique_name, start_fragment, end_fragment))
                    current_article_annotations.append((technique_name, set(range(start_fragment, end_fragment))))
            start_fragment += random.randint(0, max(1, article_length-start_fragment))
            end_fragment = min(start_fragment + random.randint(1,25), article_length)
        print("article %s: added %d fragments" % (article_id, len(current_article_annotations)))    

