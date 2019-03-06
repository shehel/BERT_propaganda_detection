
import argparse
import logging
import re
import os.path

regex = re.compile("article([0-9]+).*")

SUFFIX=".task2.labels"

logging.basicConfig(level=logging.DEBUG)


def read_input_file(filename):

    lines = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            lines.append(line)
    return lines


def load_gold_file(span_file):

    gold_set = []
    with open(span_file, "r") as f:
        for line in f.readlines():
            start,end = line.rstrip().split("\t")[2:4]
            gold_set.append(set(range(int(start),int(end))))
        #spans = [line.rstrip().split("\t")[2:4] for line in f.readlines()]
    #return spans
    return gold_set


def get_article_id_from_file_name(filename):

    return filename.split(".")[0]


def extract_sentence_labels(lines, gold_set):

    labels = []
    span_starts, span_ends = ([], [])
    for i, l in enumerate(lines):
        if i==0:
            span_starts.append(0)
        else:
            span_starts.append(span_ends[i-1])
        span_ends.append(span_starts[i]+len(l))

    i=1
    for span_start, span_end in zip(span_starts, span_ends):
        span_curr_sentence = set(range(span_start, span_end))
        found = False
        for annotation_set in gold_set:
            if len(annotation_set.intersection(span_curr_sentence))>0:
                logging.debug("row %d spans from [%d,%d), which is included in the span of one annotation: %s" % (i, span_start, span_end, str(annotation_set)))
                labels.append("propaganda")
                found = True
                break
        if not found:
            logging.debug("row %d spans from [%d,%d), it does not intersect with any annotation span" % (i, span_start, span_end))
            labels.append("non-propaganda")
        i+=1

    return labels


def main(args):
    logging.info("Processing input file %s\nand gold file %s\n"%(args.input, args.gold_file))
    lines = read_input_file(args.input)
    spans = load_gold_file(args.gold_file)
    labels = extract_sentence_labels(lines, spans)
    if len(lines) != len(labels):
        logging.error("files %s %s number of lines: %d, %d" % (args.input, args.gold_file, len(lines), len(labels)))
    with open(args.output, "w") as fout:
        for i, l in enumerate(labels):
            fname = regex.match(os.path.basename(args.input)).group(1) #args.input.split("/")[2][7:]
            fout.write("%s\t%d\t%s\n"%(fname, i+1, l)) #labels[i]))
    logging.info("Created output file " + args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create the gold file for task 2 from the gold file for task 3.")
    parser.add_argument('-i', '--input-file', dest='input', required=True, help="File with news article divided by sentences")
    parser.add_argument('-g', '--gold', dest='gold_file', required=True, help="Task 3 Gold label file (plain text)")
    parser.add_argument('-t', '--span-type', dest='span_type', required=False, default="char", help="Span type: computed at char or token level")
    parser.add_argument('-o', '--output', dest='output', required=False, help="Output file. The output file will be '[input]."+ SUFFIX +" if nothing is provided")
    arguments = parser.parse_args()
    main(arguments)
