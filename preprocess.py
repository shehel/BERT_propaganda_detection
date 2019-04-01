import logging
from utils import *
import pandas as pd
import argparse
logging.basicConfig(level=logging.INFO)

def main(args):
    prop_tech_e, prop_tech, _, _, p2id = settings(args.techniques, args.binary, args.bio)
    ids, texts, labels = read_data(args.dataset, binary=args.binary)
    logging.info("Data read")
    
    flat_list_i, flat_list, flat_list_l, _ = corpus2list(p2id, ids, texts, labels, args.binary, args.bio)
    if args.bio == None:
        df = {"ID":flat_list_i, "Tokens":flat_list, "Labels": flat_list_l}
    else: 
        #encoded = bio_encoding(flat_list_l)
        bio = []
        bio_l = []
        bio_ids = []
        count = 1
        prev = flat_list_i[0]
        for i,x,y in zip(flat_list_i, flat_list, flat_list_l):
            if i != prev:
                count = 1
            for token, label in zip(x, y):
                bio_ids.append(str(i)+'_'+str(count))
                bio.append(token)
                bio_l.append(prop_tech_e[int(label)])
            bio_ids.append('')
            bio.append('')
            bio_l.append('')
            count = count + 1
            prev = i
        
        df = {"Token":bio, "Label": bio_l}
        logging.info("Data in BIO Format")

    df = pd.DataFrame(df)

    ds = args.output
    df.to_csv(ds, index=False, header=None, sep='\t')
    
    logging.info("Dataset written to %s" % (ds))

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Preprocessing step to obtain model compatible inputs")
    parser.add_argument('-d', '--dataset-dir', dest='dataset', required=True, help="Directory containing the articles and labels.")
    parser.add_argument('-o', '--output-file', dest='output', required=True, help="Name of the file to store output to")
    parser.add_argument('-s', '--binary', dest='binary', required=False, help="Provide the name of the label if the task is binary classification")
    parser.add_argument('-b', '--bio', dest='bio', required=False, help="Use this flag to get output in coNLL-2002 format", type=bool, nargs="?", const=True)
    parser.add_argument('-t', '--techniques', dest='techniques', required=False, help="Location of the propaganda techniques file", type=str, default="tools/data/propaganda-techniques-names.txt") 
    main(parser.parse_args())
