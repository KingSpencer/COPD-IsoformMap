#!/bin/bash
#i=0
#python Training_keras.py --nodes $i --epoch 100 --lr 1e-5 --cross_val 5
#for i in {1..9}
#do
# features 255124
# features 22020
#i="256 128 64 16"
for i in {1..6}
do
    echo "Doing $i-th node search!"
    python Training_keras.py \
        --data_dir "/home/zifeng/Research/COPD/data/data_last" \
        --exp_name "lsearch_last" \
        --nodes "128 64 $((2**i))" \
        --choice "gene" \
        --model_type mlp \
        --epoch 25 \
        --batch_size 256 \
        --lr 3e-4 \
        --l1 1e-5 \
        --dropout 0 \
        --cross_val 5 \
        --feature_map True \
        --map_dir "data/mapping_data_last" \
        --save_log "./last_results/last_data_search_gene.txt" \
        --map_layer False \
        --fs_layer False \
        --trans_supervision False \
        --weights_stats True \
        --epsilon 0.02 \
        --emph False
done
