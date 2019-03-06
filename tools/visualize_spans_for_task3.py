#!/usr/bin/python

import sys

if len(sys.argv) < 3:
   sys.exit("Usage: %s <spans tsv file> <input text 1>"%(sys.argv[0]))

span_file = sys.argv[1]
file1 = sys.argv[2]

with open(span_file, "r") as f:
   spans = [ line.rstrip().split("\t")[0:4] for line in f.readlines() ]

#with open(file1, "r", encoding="latin-1") as f:
with open(file1, "r") as f:
   s1 = f.read()
   for doc, label, start, end in spans:
       print("%s\t%s\t%s\t%s\t%s" % (doc,label,start,end,s1[int(start):int(end)]))
