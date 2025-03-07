DATA_PATH=./data/saliency/pretrain/

python ./data_generation/pretrain/gen_color.py \
    --main_path $DATA_PATH \
    --total_images 300 \
    --images_per_tar 10 \
    --tokenizer_model gpt2

python ./data_generation/pretrain/gen_orientation.py \
    --main_path $DATA_PATH \
    --total_images 300 \
    --images_per_tar 10 \
    --tokenizer_model gpt2

python ./data_generation/pretrain/gen_size.py \
    --main_path $DATA_PATH \
    --total_images 300 \
    --images_per_tar 10 \
    --tokenizer_model gpt2