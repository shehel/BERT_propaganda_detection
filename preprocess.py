import logging
import pickle
from utils import *
import pandas as pd
import argparse
logging.basicConfig(level=logging.INFO)
def read_from_dir(path: str, binary: str, test: bool, p2id: dict(), bio: bool) -> list:
    
    if not bio:
        # df = {"ID":flat_list_i, "Tokens":flat_list, "Labels": flat_list_l}
        if not test:
            dataset = corpus2list(p2id, ids, texts, labels, binary, bio)
        else:
            dataset = test2list(ids, texts)
        
        dataset = {"id":dataset[0], "sentences":dataset[1], "label":dataset[2], "spacy":dataset[3]}

    else: 
        flat_list_i, flat_list, flat_list_l, flat_list_s = corpus2list(p2id, ids, texts, labels, args.binary, args.bio)
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
        
            dataset = {"token":bio, "label":bio_l}
        logging.info("Data in BIO Format")
    logging.info("Data read")
    return dataset
def main(args):
    #prop_tech_e, prop_tech, _, _, p2id = settings(args.techniques, args.binary, args.bio)
    print (args.test)
    ids, texts, labels = read_data(args.dataset, args.test, args.binary)
    #dataset = read_from_dir(args.dataset, args.binary, args.test, p2id, args.bio)
    # df = pd.DataFrame(df)
    ds = {"ID":ids, "Text": texts, "Label": labels}
    with open(args.output, 'wb') as handle:
        pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #df.to_csv(ds, index=False, header=None, sep='\t')
    
    logging.info("Dataset written to %s" % (args.output))

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Preprocessing step to obtain model compatible inputs")
    parser.add_argument('-l', '--test-data', dest='test', required=False, help="Use this flag to specify if it's not a test set in which case, labels will be returned.", type=bool, nargs="?", const=True)
    parser.add_argument('-d', '--dataset-dir', dest='dataset', required=True, help="Directory containing the articles and labels.")
    parser.add_argument('-o', '--output-file', dest='output', required=True, help="Name of the file to store output to.")
    parser.add_argument('-s', '--binary', dest='binary', required=False, help="Provide the name of the label if the task is binary classification.")
    #parser.add_argument('-t', '--techniques', dest='techniques', required=False, help="Location of the propaganda techniques file.", type=str, default="tools/data/propaganda-techniques-names.txt") 
    main(parser.parse_args())
