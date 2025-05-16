#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path /data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/llavanext-siglip-Qwen2.5-0.5B-Instruct-mlp2x_gelu-pretrain_PathResVL_2dms_instruct \
    --question-file \
    /data1/trinh/code/ViLa/quilt-llava/playground/data/eval/pmcvqa/pmcvqa_test_wo_ans.json \
    --image-folder \
    /ssd1/trinh/data/ViLa/images/quilt_instruct/Quilt_LLaVA/QuiltVQA_RED/pmcvqa_images \
    --answers-file \
    /data1/trinh/code/ViLa/quilt-llava/playground/data/eval/pmcvqa/answers/llavanext-siglip-Qwen2.5-0.5B-Instruct-mlp2x_gelu-pretrain_PathResVL_2dms_instruct.jsonl


#CKPT="wisdomik/Quilt-Llava-v1.5-7b" # or "./checkpoints/...your model"
#CKPT_model="/data1/trinh/code/ViLa/image_text/VILA_0820/checkpoints/8b/vila-v1.5-mm-pretrain" # first argument is the model path #viim
#CKPT="8b-vila-v1.5-mm-pretrain"

#CKPT_model="/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/llavanext-siglip-Qwen2.5-0.5B-Instruct-mlp2x_gelu-pretrain_PathResVL_2dms_instruct"
#CKPT="llavanext-siglip-Qwen2.5-0.5B-Instruct-mlp2x_gelu-pretrain_PathResVL_2dms_instruct"
#
#
#mkdir -p ./playground/data/eval/pmcvqa/answers
#
## PMCVQA
#python -m llava.eval.model_vqa \
#    --model-path $CKPT_model \
#    --question-file /data1/trinh/code/ViLa/quilt-llava/playground/data/eval/pmcvqa/pmcvqa_test_wo_ans.json \
#    --image-folder /ssd1/trinh/data/ViLa/images/quilt_instruct/Quilt_LLaVA/QuiltVQA_RED/pmcvqa_images \
#    --answers-file /data1/trinh/code/ViLa/quilt-llava/playground/data/eval/pmcvqa/answers/$CKPT-w-yn-pmc.jsonl

# python -W ignore llava/eval/run_vila.py     --model-path /data1/trinh/code/ViLa/image_text/VILA_0820/checkpoints/8b/vila-v1.5-mm-pretrain    --conv-mode llama_3     --query "<image>\nDescribe the key features observed in the histopathological image."     --image-file "/data1/trinh/code/ViLa/image_text/VILA_0820/scripts/v1_5/eval/eval_quilt/a/4.jpeg"
# Evaluate
# PMCVQA
#python llava/eval/pmc_eval.py \
#    --question-file /data1/trinh/code/ViLa/quilt-llava/playground/data/eval/pmcvqa/pmcvqa_test_wo_ans.json \
#    --result-file /data1/trinh/code/ViLa/quilt-llava/playground/data/eval/pmcvqa/answers/$CKPT-w-yn-pmc.jsonl \
#    --output-file /data1/trinh/code/ViLa/quilt-llava/playground/data/eval/pmcvqa/answers/$CKPT_output.jsonl \
#    --output-result /data1/trinh/code/ViLa/quilt-llava/playground/data/eval/pmcvqa/answers/$CKPT_result.json
