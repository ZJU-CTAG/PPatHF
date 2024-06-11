set -x

MODEL_PATH="xxx"  # the pretrained LLM weights
DATA_PATH="xxx"  # the dataset used for finetuning
OUTDIR="xxx"  # the output of the peft model

if [ -d "$OUTDIR" ]; then
  echo "The directory $OUTDIR already exists. Do you want to continue? (y/n)"
  read answer
  if [ "$answer" = "n" ]; then
    echo "The script has been terminated."
    exit 1
  fi
fi

mkdir -p "$OUTDIR/logs/"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=9999 train.py \
    --output_dir ${OUTDIR} \
    --model_name_or_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --model_max_length 4096 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --int8 True \
    --fp16 False \
    --bf16 True \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --group_by_length False \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 20 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0. \
    --logging_steps 1 \
    --log_level "info" \
    --report_to "tensorboard" \
    --tf32 True 2>&1 | tee "${OUTDIR}/logs/debug.log"
