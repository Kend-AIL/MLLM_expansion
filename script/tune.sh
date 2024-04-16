DATA_PATH=/workspace/data/imaterialist-fashion-2020-fgvc7/train_dataset.json
IMAGE_PATH=/workspace/data/imaterialist-fashion-2020-fgvc7


deepspeed --include localhost:4,5,6,7 /workspace/expansion_MLLM/expand_llava/train/train.py \
    --deepspeed /workspace/new_train/zero2.json \
    --model_name_or_path /workspace/imp \
    --version plain \
    --data_path  $DATA_PATH\
    --image_folder $IMAGE_PATH \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --tune_mm_mlp_adapter True \
    --tune_vision_tower True\
    --tune_vit_from_layer -1 \
    --fp16 True \
    --output_dir /workspace/test_save \
    --num_train_epochs 8\
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --tf32 False \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 15 \
    --lazy_preprocess True \
    --report_to tensorboard \
    