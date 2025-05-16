#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
#CKPT="wisdomik/Quilt-Llava-v1.5-7b" # or "./checkpoints/...your model"
CKPT="/ssd1/trinh/data/ViLa/images/quilt_instruct/Quilt-Llava-v1.5-7b" # or "./checkpoints/...your model"
CKPT_name="Quilt-Llava-v1.5-7b" # or "./checkpoints/...your model"

mkdir -p ./playground/data/eval/quiltvqa/answers

# QUILT-VQA
python -m llava.eval.model_vqa \
    --model-path $CKPT \
    --question-file /ssd1/trinh/data/ViLa/images/quilt_instruct/Quilt_LLaVA/Quilt_VQA/quiltvqa_test_wo_ans.jsonl \
    --image-folder /ssd1/trinh/data/ViLa/images/quilt_instruct/Quilt_LLaVA/Quilt_VQA/quilt_vqa/images \
    --answers-file /data1/trinh/code/ViLa/quilt-llava/playground/data/eval/quiltvqa/answers/$CKPT_name-w-yn-quilt.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# Evaluate
# QUILT-VQA
python llava/eval/quilt_eval.py \
    --quilt True \
    --gt /ssd1/trinh/data/ViLa/images/quilt_instruct/Quilt_LLaVA/Quilt_VQA/quiltvqa_test_w_ans.json \
    --pred /data1/trinh/code/ViLa/quilt-llava/playground/data/eval/quiltvqa/answers/$CKPT_name-w-yn-quilt.jsonl

#('Metric               Performance\n'
 # '-----------------  -------------\n'
 # 'exact match score        10.7162\n'
 # 'f1 score                 29.0258\n'
 # 'precision                20.3489\n'
 # 'recall                   62.6785\n'
 # 'yes/no accuracy          40.5248')
