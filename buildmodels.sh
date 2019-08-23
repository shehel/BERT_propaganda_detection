# Script to create processed intermediate dataset files 
# for each propaganda technique in binary classification
# setting. Scripts takes a file containing all the relevant
# propaganda technique separated by line breaks as argument.
while IFS='' read -r line || [[ -n "$line" ]]; do
    echo "Starting preprocessing: $line"
    python preprocess.py -d ./train-split/tasks-2-3/train-train/ -o pro_data/"$line".train -s $line -line
    python preprocess.py -d ./train-split/tasks-2-3/train-dev/ -o pro_data/"$line".dev -s $line -l
    #python train.py --expID "$line"_10EBCF --trainDataset datasets/"$line"_train.csv --valDataset datasets/"$line"_dev.csv --model bert-base-cased --LR 3e-5 --trainBatch 32 --nEpochs 10 --classType single_class --binaryLabel $line --nLabels 4
done < "$1"