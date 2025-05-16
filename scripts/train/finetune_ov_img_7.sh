export OMP_NUM_THREADS=8
#export NCCL_IB_DISABLE=0
#export NCCL_IB_GID_INDEX=3
#export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

#export NCCL_P2P_DISABLE=1
#export NCCL_SHM_DISABLE=1


#LLM_VERSION="/data1/trinh/code/ViLa/video/LLaVA-NeXT/hf_weight/Qwen/Qwen2.5-0.5B-Instruct"
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
NUM_GPUS=4
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ADDR="127.0.0.1"  # or another valid IP address/hostname
echo "MASTER_ADDR="$ADDR
############### Pretrain ################

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-Qwen2.5-7B-Instruct-mlp2x_gelu-pretrain_PathResVL_pretrain_bs4"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

################ Finetune ################
# Stage 2
PROMPT_VERSION="qwen_1_5"
#RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_PathResVL_instruct_SC_video_v0_bs2_from_im+clip"
RUN_NAME="llava-si-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_PathResVL_instruct_stage2_img_bs1_from_stage1_im"
#PREV_STAGE_CHECKPOINT="/mnt/bn/vl-research/checkpoints/onevision/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mid_to_final_next_3m_am9_july14" # replace it with your last checkpoint training from single image collection
PREV_STAGE_CHECKPOINT="/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/llavanext-siglip-Qwen2.5-7B-Instruct-mlp2x_gelu-pretrain_PathResVL_2dms_phase2_instruct" # replace it with your last checkpoint training from single image collection
#PREV_STAGE_CHECKPOINT="/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/onevision/llava-onevision-siglip-Qwen2.5-7B-Instruct-ov_PathResVL_instruct_video_chunk_yolo_gpt4o_train_cls33_all+2dms-anyres_bs2" # replace it with your last checkpoint training from single image collection
#PREV_STAGE_CHECKPOINT="/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/onevision/llava-onevision-siglip-Qwen2.5-7B-Instruct-ov_PathResVL_instruct_video_chunk_yolo_gpt4o_train_cls33_all_bs2" # replace it with your last checkpoint training from single image collection
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

##    --data_path /data1/trinh/code/ViLa/video/LLaVA-NeXT/scripts/train/onevision_s_video.yaml \

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path /data1/trinh/code/ViLa/video/LLaVA-NeXT/scripts/train/onevision_stage2_image.yaml \
    --image_folder /ssd1/trinh/data/ViLa/images/ \
    --video_folder /ssd1/trinh/raw_data/video_dataset_train/video_one_cases_train/ \
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
    --output_dir ./checkpoints/onevision_sc/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
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
################# Finetune ################################ Finetune ################
## Stage 2
#PROMPT_VERSION="qwen_1_5"
#RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_PathResVL_instruct_video_chunk_yolo_gpt4o_train_cls33_all_bs2"
##PREV_STAGE_CHECKPOINT="/mnt/bn/vl-research/checkpoints/onevision/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mid_to_final_next_3m_am9_july14" # replace it with your last checkpoint training from single image collection
#PREV_STAGE_CHECKPOINT="/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/llavanext-siglip-Qwen2.5-7B-Instruct-mlp2x_gelu-pretrain_PathResVL_2dms_phase2_instruct" # replace it with your last checkpoint training from single image collection
#echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
#echo "MID_RUN_NAME: ${RUN_NAME}"
#
#ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
#    llava/train/train_mem.py \
#    --deepspeed scripts/zero3.json \
#    --model_name_or_path $PREV_STAGE_CHECKPOINT \
#    --version $PROMPT_VERSION \
#    --data_path /data1/trinh/code/ViLa/video/LLaVA-NeXT/scripts/train/onevision_video.yaml \
#    --image_folder /ssd1/trinh/data/ViLa/images/ \
#    --video_folder /data1/trinh/data/raw_data/quilt1m_data/clips_histo_frame_clean \
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
#    --run_name $RUN_NAME \
#    --output_dir ./checkpoints/onevision/$RUN_NAME \
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
################# Finetune ################
#
## Stage 2
#PROMPT_VERSION="qwen_1_5"
#RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_PathResVL_instruct_video_chunk_yolo_gpt4o_train_cls33_datav0"
##PREV_STAGE_CHECKPOINT="/mnt/bn/vl-research/checkpoints/onevision/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mid_to_final_next_3m_am9_july14" # replace it with your last checkpoint training from single image collection
#PREV_STAGE_CHECKPOINT="/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/llavanext-siglip-Qwen2.5-0.5B-Instruct-mlp2x_gelu-pretrain_PathResVL_2dms_instruct" # replace it with your last checkpoint training from single image collection
#echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
#echo "MID_RUN_NAME: ${RUN_NAME}"
#
#ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
#    llava/train/train_mem.py \
#    --deepspeed scripts/zero3.json \
#    --model_name_or_path $PREV_STAGE_CHECKPOINT \
#    --version $PROMPT_VERSION \
#    --data_path /data1/trinh/code/ViLa/video/LLaVA-NeXT/scripts/train/onevision_video.yaml \
#    --image_folder /ssd1/trinh/data/ViLa/images/ \
#    --video_folder /data1/trinh/data/raw_data/quilt1m_data/clips_histo_frame_clean \
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
#    --run_name $RUN_NAME \
#    --output_dir ./checkpoints/onevision/$RUN_NAME \
#    --num_train_epochs 1 \
#    --per_device_train_batch_size 4 \
#    --per_device_eval_batch_size 4 \
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

#    --video_center_crop 0.5 \


# You can delete the sdpa attn_implementation if you want to use flash attn

# | 64/8746 [12:02<26:36:10, 11.03s/it]





# #Rank 0:  Model saved to ./checkpoints/onevision/llava-onevision-siglip-Qwen2.5-0.5B-Instruct-ov_PathResVL_instruct_video_chunk0_gpt4o_train_cls33_datav0
# 11 hours
#{'loss': 1.9897, 'grad_norm': 5.1166581246243785, 'learning_rate': 0.0, 'epoch': 1.0}
 #{'train_runtime': 40083.3485, 'train_samples_per_second': 1.741, 'train_steps_per_second': 0.054, 'train_loss': 1.947293723285745, 'epoch': 1.0}
 #100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2180/2180 [11:08:02<00:00, 18.39s/it]
 #Rank 0:  Only save projectors: False
 #Rank 0:  Model saved to ./checkpoints/onevision/llava-onevision-siglip-Qwen2.5-0.5B-Instruct-ov_PathResVL_instruct_video_chunk0_gpt4o_train_cls33_datav0
 #wandb: 🚀 View run llava-onevision-siglip-Qwen2.5-0.5B-Instruct-ov_PathResVL_instruct_video_chunk0_gpt4o_train_cls33_datav0 at: https://wandb.ai/timmyvg/huggingface/runs/lacedr1g
 #wandb: Find logs at: wandb/run-20240929_055534-lacedr1g/logs

#  all data
# llava-onevision-siglip-Qwen2.5-0.5B-Instruct-ov_PathResVL_instruct_video_chunk0_gpt4o_train_cls33


#{'loss': 1.9055, 'grad_norm': 5.067855586621875, 'learning_rate': 1.3802819178398985e-10, 'epoch': 1.0}
 #{'loss': 1.8937, 'grad_norm': 4.817477100379504, 'learning_rate': 8.833818906039959e-11, 'epoch': 1.0}
 #{'loss': 1.9436, 'grad_norm': 5.065754091792606, 'learning_rate': 4.969029536061598e-11, 'epoch': 1.0}
 #{'loss': 1.8628, 'grad_norm': 5.1166067286001224, 'learning_rate': 2.2084596038030036e-11, 'epoch': 1.0}
 #{'loss': 1.8985, 'grad_norm': 5.188634207229866, 'learning_rate': 5.5211520577636015e-12, 'epoch': 1.0}
 #{'loss': 1.9906, 'grad_norm': 5.110677207482676, 'learning_rate': 0.0, 'epoch': 1.0}
 #{'train_runtime': 42812.181, 'train_samples_per_second': 1.63, 'train_steps_per_second': 0.051, 'train_loss': 1.9473023125337898, 'epoch': 1.0}
 #100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2180/2180 [11:53:30<00:00, 19.64s/it]
 #Rank 0:  Only save projectors: False
 #Rank 0:  Model saved to ./checkpoints/onevision/llava-onevision-siglip-Qwen2.5-0.5B-Instruct-ov_PathResVL_instruct_video_chunk0_gpt4o_train_cls33_datav0_video_cc05
 #wandb: 🚀 View run llava-onevision-siglip-Qwen2.5-0.5B-Instruct-ov_PathResVL_instruct_video_chunk0_gpt4o_train_cls33_datav0_video_cc05 at: https://wandb.ai/timmyvg/huggingface/runs/dadnwobm
 #wandb: Find logs at: wandb/run-20240930_063939-dadnwobm/logs


#
# {'loss': 1.9522, 'grad_norm': 5.189381456285028, 'learning_rate': 8.833818906039959e-11, 'epoch': 1.0}
#{'loss': 1.9435, 'grad_norm': 4.787241459519883, 'learning_rate': 4.969029536061598e-11, 'epoch': 1.0}
#{'loss': 1.8772, 'grad_norm': 5.20646740234398, 'learning_rate': 2.2084596038030036e-11, 'epoch': 1.0}
#{'loss': 1.8625, 'grad_norm': 4.973339191912084, 'learning_rate': 5.5211520577636015e-12, 'epoch': 1.0}
#{'loss': 1.9419, 'grad_norm': 4.994383021633049, 'learning_rate': 0.0, 'epoch': 1.0}
#{'train_runtime': 44233.4769, 'train_samples_per_second': 1.577, 'train_steps_per_second': 0.049, 'train_loss': 1.9473338397817874, 'epoch': 1.0}
#100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2180/2180 [12:17:12<00:00, 20.29s/it]
#Rank 0:  Only save projectors: False
#Rank 0:  Model saved to ./checkpoints/onevision/llava-onevision-siglip-Qwen2.5-0.5B-Instruct-ov_PathResVL_instruct_video_chunk_yolo_gpt4o_train_cls33_datav0_video_cc05
#wandb: 🚀 View run llava-onevision-siglip-Qwen2.5-0.5B-Instruct-ov_PathResVL_instruct_video_chunk_yolo_gpt4o_train_cls33_datav0_video_cc05 at: https://wandb.ai/timmyvg/huggingface/runs/bckhgqfu
#


#llava-onevision-siglip-Qwen2.5-0.5B-Instruct-ov_PathResVL_instruct_video_chunk_yolo_gpt4o_train_cls33_all_error_05_7
#{'loss': 1.6451, 'grad_norm': 3.8502877776976474, 'learning_rate': 1.3718391389527796e-12, 'epoch': 1.0}
 #{'loss': 1.7293, 'grad_norm': 3.89604628509537, 'learning_rate': 0.0, 'epoch': 1.0}
 #{'train_runtime': 68915.7845, 'train_samples_per_second': 2.031, 'train_steps_per_second': 0.063, 'train_loss': 1.749001869887093, 'epoch': 1.0}
 #100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4373/4373 [19:08:34<00:00, 15.76s/it]
 #Rank 0:  Only save projectors: False
 #Rank 0:  Model saved to ./checkpoints/onevision/llava-onevision-siglip-Qwen2.5-0.5B-Instruct-ov_PathResVL_instruct_video_chunk_yolo_gpt4o_train_cls33_all
 #wandb: 🚀 View run llava-onevision-siglip-Qwen2.5-0.5B-Instruct-ov_PathResVL_instruct_video_chunk_yolo_gpt4o_train_cls33_all at: https://wandb.ai/timmyvg/huggingface/runs/5j4cj4sq
 #wandb: Find logs at: wandb/run-20241108_193617-5j4cj4sq/logs


#{'loss': 1.7176, 'grad_norm': 4.549929319776819, 'learning_rate': 1.2346547735297975e-11, 'epoch': 1.0}
 #{'loss': 1.6768, 'grad_norm': 4.22867639992033, 'learning_rate': 5.487355803635019e-12, 'epoch': 1.0}
 #{'loss': 1.6457, 'grad_norm': 3.8497005951784398, 'learning_rate': 1.3718391389527796e-12, 'epoch': 1.0}
 #{'loss': 1.7281, 'grad_norm': 3.89093599785132, 'learning_rate': 0.0, 'epoch': 1.0}
 #{'train_runtime': 69112.2901, 'train_samples_per_second': 2.025, 'train_steps_per_second': 0.063, 'train_loss': 1.7489954497049254, 'epoch': 1.0}
 #100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4373/4373 [19:11:51<00:00, 15.80s/it]
 #Rank 0:  Only save projectors: False
 #Rank 0:  Model saved to ./checkpoints/onevision/llava-onevision-siglip-Qwen2.5-7B-Instruct-ov_PathResVL_instruct_video_chunk_yolo_gpt4o_train_cls33_all_bs2
 #wandb: 🚀 View run llava-onevision-siglip-Qwen2.5-7B-Instruct-ov_PathResVL_instruct_video_chunk_yolo_gpt4o_train_cls33_all_bs2 at: https://wandb.ai/timmyvg/huggingface/runs/k78w88qk
