export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO



LLM_VERSION="Qwen/Qwen2.5-7B-Instruct"

#LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
LLM_VERSION_CLEAN=$(basename "$LLM_VERSION")

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="siglip"


############### my input ################
NNODES=1
RANK=0
PORT=25050
NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ADDR="127.0.0.1"  # or another valid IP address/hostname
echo "MASTER_ADDR="$ADDR
############### Pretrain ################

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-Qwen2.5-7B-Instruct-mlp2x_gelu-pretrain_PathResVL_pretrain_bs4"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

################ Finetune ################
# Stage 2
PROMPT_VERSION="qwen_1_5"

RUN_NAME="VidDiag_LLaVA_video"
PREV_STAGE_CHECKPOINT="./" # replace it with your last checkpoint training from single image collection


echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path /data1/trinh/code/ViLa/video/LLaVA-NeXT/scripts/train/onevision_sc_video.yaml \
    --image_folder /ssd1/trinh/data/ViLa/images/ \
    --video_folder /ssd1/trinh/raw_data/pvideo/ \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir ./checkpoints/onevision_sc_v1.0/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --lora_enable True --lora_r 128 --lora_alpha 256

exit 0;
