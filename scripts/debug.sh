CUDA_VISIBLE_DEVICES=0 python3 groma/train/train.py \
--llm checkpoints/vicuna-7b-v1.5 \
--perceiver checkpoints/ddetr_box \
--dataset_config groma/data/configs/vl_pretrain.py \
--freeze_perceiver True \
--freeze_llm True \
--bf16 True \
--tf32 True \
--output_dir ./checkpoints/debug \
--num_train_epochs 0.001 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2500 \
--save_total_limit 1 \
--learning_rate 1e-4 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--model_max_length 2048 \
--report_to none