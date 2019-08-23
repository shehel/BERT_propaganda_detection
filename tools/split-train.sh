# We assume that you have downloaded the input data
# and that you have unzipped it in datasets (you can also edit $DATA_DIR)

DATA_DIR=datasets # training set articles are supposed to be in folder $DATA_DIR/train-articles
NUMBER_OF_ARTICLES_IN_TRAIN_TRAIN_PARTITION=245 # 70% of the training set articles will be in train-train
let "NUMBER_OF_ARTICLES_IN_TRAIN_DEV_PARTITION = `ls -1 $DATA_DIR/train-articles/*.txt | wc -l` - $NUMBER_OF_ARTICLES_IN_TRAIN_TRAIN_PARTITION"
RANDOM_SOURCE=$0 # seed for random splitting of articles into train-train and train-dev (change it to modify the splitting)

# The data split will be saved in TRAIN_TRAIN_SPLIT_DIR and TRAIN_DEV_SPLIT_DIR
TRAIN_TRAIN_SPLIT_DIR=${DATA_DIR}/train-train
TRAIN_DEV_SPLIT_DIR=${DATA_DIR}/train-dev

echo "Splitting folders $DATA_DIR/train-articles, $DATA_DIR/train-labels-SLC, $DATA_DIR/train-labels-FLC into two subsets"

# Creating output folders or deleting existing articles in them
for d in "${TRAIN_TRAIN_SPLIT_DIR}-articles" "${TRAIN_TRAIN_SPLIT_DIR}-labels-SLC" "${TRAIN_TRAIN_SPLIT_DIR}-labels-FLC" "${TRAIN_DEV_SPLIT_DIR}-articles" "${TRAIN_DEV_SPLIT_DIR}-labels-SLC" "${TRAIN_DEV_SPLIT_DIR}-labels-FLC"; do
    rm $d/* 2>/dev/null || mkdir -p $d
done

# Get article list
article_list=`find $DATA_DIR/train-articles -name 'article*.txt' -exec basename {} \; | sed 's/article//g' | sed 's/.txt//g' | sort -R --random-source $RANDOM_SOURCE`

# Saving articles to train-train
for i in `echo $article_list | tr " " "\n" | head -n $NUMBER_OF_ARTICLES_IN_TRAIN_TRAIN_PARTITION`; do
    
    cp $DATA_DIR/train-articles/article${i}.txt ${TRAIN_TRAIN_SPLIT_DIR}-articles/
    cp $DATA_DIR/train-labels-SLC/article${i}.task-SLC.labels ${TRAIN_TRAIN_SPLIT_DIR}-labels-SLC/
    cp $DATA_DIR/train-labels-FLC/article${i}.task-FLC.labels ${TRAIN_TRAIN_SPLIT_DIR}-labels-FLC/

done

# Saving articles to train-dev
for i in `echo $article_list | tr " " "\n" | tail -n $NUMBER_OF_ARTICLES_IN_TRAIN_DEV_PARTITION`; do
    
    cp $DATA_DIR/train-articles/article${i}.txt ${TRAIN_DEV_SPLIT_DIR}-articles/
    cp $DATA_DIR/train-labels-SLC/article${i}.task-SLC.labels ${TRAIN_DEV_SPLIT_DIR}-labels-SLC/
    cp $DATA_DIR/train-labels-FLC/article${i}.task-FLC.labels ${TRAIN_DEV_SPLIT_DIR}-labels-FLC/

done

# Printing output stats
for d in "${TRAIN_TRAIN_SPLIT_DIR}-articles" "${TRAIN_TRAIN_SPLIT_DIR}-labels-SLC" "${TRAIN_TRAIN_SPLIT_DIR}-labels-FLC" "${TRAIN_DEV_SPLIT_DIR}-articles" "${TRAIN_DEV_SPLIT_DIR}-labels-SLC" "${TRAIN_DEV_SPLIT_DIR}-labels-FLC"; do

    echo "created output folder $d ("`ls -1 $d | wc -l`" articles )"

done
