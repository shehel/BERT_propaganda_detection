while IFS='' read -r line || [[ -n "$line" ]]; do
    echo "Starting predictions: $line"
    python predict.py --testDataset datasets-v5/tasks-2-3/dev/ --model bert-base-cased --validBatch 64 --loadModel exp/single_class/"$line"_10EBCF/best_model.pth --outputFile bert-base-best-val/"$line"_pred.csv --classType single_class --nLabels 4 --binaryLabel $line
    #cat "$line" >> check.txt
    cat bert-base-best-val/"$line"_pred.csv >> merged_10EBCF.csv
done < "$1"
