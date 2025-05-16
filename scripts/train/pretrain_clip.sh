export OMP_NUM_THREADS=8
#export NCCL_IB_DISABLE=0
#export NCCL_IB_GID_INDEX=3
#export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

#LLM_VERSION="Qwen/Qwen2-7B-Instruct"
#LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
#VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
#VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

#LLM_VERSION="Qwen/Qwen2-7B-Instruct"
#LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
#VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
#VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

LLM_VERSION="/data1/trinh/code/ViLa/video/LLaVA-NeXT/hf_weight/Qwen/Qwen2.5-0.5B-Instruct"
LLM_VERSION_CLEAN=$(basename "$LLM_VERSION")
VISION_MODEL_VERSION="/data1/trinh/code/ViLa/weight/vision/QuiltNet-B-32/"
VISION_MODEL_VERSION_CLEAN=$(basename "$VISION_MODEL_VERSION")

############### my input ################
NNODES=1
RANK=0
PORT=25051
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ADDR="127.0.0.1"  # or another valid IP address/hostname
echo "MASTER_ADDR="$ADDR
############### Pretrain ################

PROMPT_VERSION=plain

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_PathResVL_pretrain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /ssd1/trinh/data/ViLa/images/PathResVL_pretrain.json \
    --image_folder /ssd1/trinh/data/ViLa/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME \
    --attn_implementation sdpa

# You can delete the sdpa attn_implementation if you want to use flash attn
