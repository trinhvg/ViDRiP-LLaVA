export OMP_NUM_THREADS=8
#export NCCL_IB_DISABLE=0
#export NCCL_IB_GID_INDEX=3
#export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

#export NCCL_P2P_DISABLE=1
#export NCCL_SHM_DISABLE=1


#LLM_VERSION="/data1/trinh/code/ViLa/video/LLaVA-NeXT/hf_weight/Qwen/Qwen2.5-7B-Instruct"
LLM_VERSION="Qwen/Qwen2.5-7B-Instruct"

#LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
LLM_VERSION_CLEAN=$(basename "$LLM_VERSION")

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
#VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
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

#llavanext-google_siglip-so400m-patch14-384-_data1_trinh_code_ViLa_video_LLaVA-NeXT_hf_weight_Qwen_Qwen2.5-0.5B-Instruct-mlp2x_gelu-pretrain_PathResVL_pretrain
#llavanext-siglip-Qwen2.5-0.5B-Instruct-mlp2x_gelu-pretrain_PathResVL_pretrain

PROMPT_VERSION="qwen_1_5"

#BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-Qwen2.5-7B-Instruct-mlp2x_gelu-pretrain_PathResVL_pretrain_bs4"
#MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-instruct_SC_video_v0_bs1_from_stage_0"


BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-Qwen2.5-7B-Instruct-mlp2x_gelu-pretrain_PathResVL_pretrain_bs4"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-_instruct_SC_video_v1.0_gpt4o_mini_diagnosis_short_yolo_bs2_from_stage_0"


#BASE_RUN_NAME_01="llavanext-siglip-Qwen2.5-0.5B-Instruct-mlp2x_gelu-pretrain_PathResVL_pretrain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

CKPT_PATH=$LLM_VERSION # this could also be the previous stage checkpoint

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path /data1/trinh/code/ViLa/video/LLaVA-NeXT/scripts/train/onevision_sc_video.yaml \
    --image_folder /ssd1/trinh/data/ViLa/images/ \
    --video_folder /ssd1/trinh/raw_data/pvideo/ \
    --pretrain_mm_mlp_adapter="./checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir ./checkpoints/onevision_sc_v1.0/${MID_RUN_NAME} \
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
    --frames_upbound 32
exit 0;

############### Pretrain ################

##llavanext-google_siglip-so400m-patch14-384-_data1_trinh_code_ViLa_video_LLaVA-NeXT_hf_weight_Qwen_Qwen2.5-0.5B-Instruct-mlp2x_gelu-pretrain_PathResVL_pretrain
##llavanext-siglip-Qwen2.5-0.5B-Instruct-mlp2x_gelu-pretrain_PathResVL_pretrain
#
#PROMPT_VERSION="qwen_1_5"
#
##BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-Qwen2.5-7B-Instruct-mlp2x_gelu-pretrain_PathResVL_pretrain_bs4"
##MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-instruct_SC_video_v0_bs1_from_stage_0"
#
#
#BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-Qwen2.5-7B-Instruct-mlp2x_gelu-pretrain_PathResVL_pretrain_bs4"
#MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-instruct_SC_video_v0_bs1_from_stage_0"
#
#
##BASE_RUN_NAME_01="llavanext-siglip-Qwen2.5-0.5B-Instruct-mlp2x_gelu-pretrain_PathResVL_pretrain"
#echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
#
#CKPT_PATH=$LLM_VERSION # this could also be the previous stage checkpoint
#
#ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
#    llava/train/train_mem.py \
#    --deepspeed scripts/zero3.json \
#    --model_name_or_path ${CKPT_PATH} \
#    --version ${PROMPT_VERSION} \
#    --data_path /data1/trinh/code/ViLa/video/LLaVA-NeXT/scripts/train/onevision_s_video.yaml \
#    --image_folder /ssd1/trinh/data/ViLa/images/ \
#    --video_folder /ssd1/trinh/raw_data/video_dataset_train/video_one_cases_train/ \
#    --pretrain_mm_mlp_adapter="./checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
#    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
#    --mm_vision_tower_lr=2e-6 \
#    --vision_tower ${VISION_MODEL_VERSION} \
#    --mm_projector_type mlp2x_gelu \
#    --mm_vision_select_layer -2 \
#    --mm_use_im_start_end False \
#    --mm_use_im_patch_token False \
#    --group_by_modality_length True \
#    --image_aspect_ratio anyres_max_9 \
#    --image_grid_pinpoints  "(1x1),...,(6x6)" \
#    --mm_patch_merge_type spatial_unpad \
#    --bf16 True \
#    --run_name $MID_RUN_NAME \
#    --output_dir ./checkpoints/onevision_sc/${MID_RUN_NAME} \
#    --num_train_epochs 1 \
#    --per_device_train_batch_size 2 \
#    --per_device_eval_batch_size 2 \
#    --gradient_accumulation_steps 2 \
#    --evaluation_strategy "no" \
#    --save_strategy "steps" \
#    --save_steps 1000 \
#    --save_total_limit 1 \
#    --learning_rate 1e-5 \
#    --weight_decay 0. \
#    --warmup_ratio 0.03 \
#    --lr_scheduler_type "cosine" \
#    --logging_steps 1 \
#    --tf32 True \
#    --model_max_length 32768 \
#    --gradient_checkpointing True \
#    --dataloader_num_workers 4 \
#    --lazy_preprocess True \
#    --report_to wandb \
#    --torch_compile True \
#    --torch_compile_backend "inductor" \
#    --dataloader_drop_last True \
#    --frames_upbound 32
#exit 0;
# You can delete the sdpa attn_implementation if you want to use flash attn
#3000

#{'loss': 0.9443, 'grad_norm': 4.6253289939032465, 'learning_rate': 0.0, 'epoch': 1.0}
 #{'train_runtime': 65389.8568, 'train_samples_per_second': 4.43, 'train_steps_per_second': 0.277, 'train_loss': 1.1655144068062948, 'epoch': 1.0}
 #100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 18105/18105 [18:09:48<00:00,  3.61s/it]
 #Rank 0:  Only save projectors: False
 #Rank 0:  Model saved to ./checkpoints/llavanext-siglip-Qwen2.5-0.5B-Instruct-mlp2x_gelu-pretrain_PathResVL_2dms_instruct
 #wandb: 🚀 View run llavanext-siglip-Qwen2.5-0.5B-Instruct-mlp2x_gelu-pretrain_PathResVL_2dms_instruct at: https://wandb.ai/timmyvg/huggingface/runs/ao9nvmwv
 #wandb: Find logs at: wandb/run-20240927_055345-ao9nvmwv/logs


#{'loss': 0.8574, 'grad_norm': 5.042614368887564, 'learning_rate': 7.667701784619397e-13, 'epoch': 1.0}
 #{'loss': 0.7232, 'grad_norm': 4.972440580128804, 'learning_rate': 3.40786751040767e-13, 'epoch': 1.0}
 #{'loss': 0.7826, 'grad_norm': 4.212061320938141, 'learning_rate': 8.5196688592859e-14, 'epoch': 1.0}
 #{'loss': 0.6779, 'grad_norm': 3.349128277506968, 'learning_rate': 0.0, 'epoch': 1.0}
 #{'train_runtime': 106200.2751, 'train_samples_per_second': 1.322, 'train_steps_per_second': 0.165, 'train_loss': 0.78129405425328, 'epoch': 1.0}
 #100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 17545/17545 [29:29:58<00:00,  6.05s/it]
 #Rank 0:  Only save projectors: False
 #Rank 0:  Model saved to ./checkpoints/llavanext-siglip-Qwen2.5-7B-Instruct-mlp2x_gelu-pretrain_PathResVL_2dms_phase2_instruct
