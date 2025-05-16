export OMP_NUM_THREADS=8
#export NCCL_IB_DISABLE=0
#export NCCL_IB_GID_INDEX=3
#export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

#export NCCL_P2P_DISABLE=1
#export NCCL_SHM_DISABLE=1


#LLM_VERSION="Qwen/Qwen2-0.5B-Instruct"
LLM_VERSION="Qwen/Qwen2.5-7B-Instruct"
LLM_VERSION_CLEAN=$(basename "$LLM_VERSION")
#LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
#VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
VISION_MODEL_VERSION_CLEAN="siglip"


############### my input ################
NNODES=1
RANK=0
PORT=25051
NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ADDR="127.0.0.1"  # or another valid IP address/hostname
echo "MASTER_ADDR="$ADDR
############### Pretrain ################

PROMPT_VERSION=plain

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_PathResVL_pretrain_bs4"
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
    --output_dir ./checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 1000 \
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




#{'loss': 2.7595, 'grad_norm': 0.3004620395255959, 'learning_rate': 1.1912350161846774e-11, 'epoch': 1.0}
 #{'loss': 3.2551, 'grad_norm': 0.3856636762883365, 'learning_rate': 2.9780875543394813e-12, 'epoch': 1.0}
 #{'loss': 2.6586, 'grad_norm': 0.3108922640575761, 'learning_rate': 0.0, 'epoch': 1.0}
 #{'train_runtime': 25512.5027, 'train_samples_per_second': 37.22, 'train_steps_per_second': 1.163, 'train_loss': 2.8163621072921816, 'epoch': 1.0}
 #100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 29675/29675 [7:05:11<00:00,  1.16it/s]
 #Rank 0:  Only save projectors: True
 #Rank 0:  Model saved to ./checkpoints/projectors/llavanext-google_siglip-so400m-patch14-384-_data1_trinh_code_ViLa_video_LLaVA-NeXT_hf_weight_Qwen_Qwen2.5-0.5B-Instruct-mlp2x_gelu-pretrain_PathResVL_pretrain


# {'train_runtime': 25713.4258, 'train_samples_per_second': 36.929, 'train_steps_per_second': 1.154, 'train_loss': 2.938936254940017, 'epoch': 1.0}
  #100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 29675/29675 [7:08:32<00:00,  1.15it/s]
  #Rank 0:  Only save projectors: True
  #Rank 0:  Model saved to ./checkpoints/projectors/llavanext-siglip-Qwen2-0.5B-Instruct-mlp2x_gelu-pretrain_PathResVL_pretrain
  #wandb: 🚀 View run llavanext-siglip-Qwen2-0.5B-Instruct-mlp2x_gelu-pretrain_PathResVL_pretrain at: https://wandb.ai/timmyvg/huggingface/runs/5gwmbsz3
  #wandb: Find logs at: wandb/run-20240927_055944-5gwmbsz3/logs


#  0%|   | 3/14838 [00:35<40:25:18,  9.81s/it]
# 30:58 3.74s/it bs4 29675
# 30:58 3.74s/it bs8 14838
#| 78/14838 [08:06<23:58:30,  5.85s/it]
