while IFS='' read -r line || [[ -n "$line" ]]; do
    echo "Starting preprocessing: $line"
    python preprocess.py -d ./datasets-v5/tasks-2-3/train/ -o datasets/"$line"_train.csv -s $line 
    python preprocess.py -d ./datasets-v5/tasks-2-3/dev/ -o datasets/"$line"_dev.csv -s $line 
    python train.py --expID "$line"_10EBCF --trainDataset datasets/"$line"_train.csv --valDataset datasets/"$line"_dev.csv --model bert-base-cased --LR 3e-5 --trainBatch 32 --nEpochs 10 --classType single_class --binaryLabel $line --nLabels 4
done < "$1"