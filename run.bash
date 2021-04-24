echo "Doing comparision experiment!"
exp='rebuttal'
python Training_keras.py \
    --data_dir "/home/zifeng/Research/COPD/data/data_last" \
    --exp_name $exp \
    --nodes "" \
    --choice "exon+transcript" \
    --model_type mlp \
    --epoch 25 \
    --batch_size 256 \
    --lr 1e-4 \
    --l1 3e-5 \
    --dropout 0 \
    --cross_val 1 \
    --feature_map True \
    --map_dir "data/mapping_data_last" \
    --save_log "./rebuttal_results/last_check_fs.txt" \
    --map_layer False \
    --fs_layer False \
    --trans_supervision False \
    --weights_stats False \
    --epsilon 0.02 \
    --emph False \
    --final_test True \
    --final_test_dir "/home/zifeng/Research/COPD/data/data_last_test" \
    --final_test_save_dir "/home/zifeng/Research/COPD/rebuttal_test_prediction/${exp}"
    # --no_train \
    # --load_model_path "/home/zifeng/Research/COPD/models/rebuttal/exon/model.h5"


