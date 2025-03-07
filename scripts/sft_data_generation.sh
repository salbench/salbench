DATA_PATH=./data/saliency/sft/

python ./data_generation/sft/gen_color.py \
    --main_path $DATA_PATH \
    --total_images 300 \
    --images_per_folder 10 \
    --tokenizer_model gpt2

python ./data_generation/sft/gen_orientation.py \
    --main_path $DATA_PATH \
    --total_images 300 \
    --images_per_folder 10 \
    --tokenizer_model gpt2

python ./data_generation/sft/gen_size.py \
    --main_path $DATA_PATH \
    --total_images 300 \
    --images_per_folder 10 \
    --tokenizer_model gpt2