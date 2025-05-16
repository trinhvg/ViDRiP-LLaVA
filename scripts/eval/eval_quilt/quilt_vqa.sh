#!/bin/bash
export CUDA_VISIBLE_DEVICES=5



#CKPT="wisdomik/Quilt-Llava-v1.5-7b" # or "./checkpoints/...your model"
#CKPT="/ssd1/trinh/data/ViLa/images/quilt_instruct/Quilt-Llava-v1.5-7b" # or "./checkpoints/...your model"
#CKPT_name="Quilt-Llava-v1.5-7b" # or "./checkpoints/...your model"
CKPT="/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/onevision_sc_v1.0/llava-onevision-siglip-Qwen2.5-7B-Instruct-ov_PathResVL_instruct_SC_video_v1.0_gpt4o_mini_diagnosis_short_yolo_bs2_from_im+clip_lora_merge"
CKPT_name="llava-onevision-siglip-Qwen2.5-7B-Instruct-ov_PathResVL_instruct_SC_video_v1.0_gpt4o_mini_diagnosis_short_yolo_bs2_from_im+clip_lora_merge" # or "./checkpoints/...your model"

mkdir -p ./playground/data/eval/quiltvqa/answers

# QUILT-VQA
python -m llava.eval.model_vqa \
    --model-path $CKPT \
    --question-file /ssd1/trinh/data/ViLa/images/quilt_instruct/Quilt_LLaVA/Quilt_VQA/quiltvqa_test_wo_ans.jsonl \
    --image-folder /ssd1/trinh/data/ViLa/images/quilt_instruct/Quilt_LLaVA/Quilt_VQA/quilt_vqa/images \
    --answers-file /data1/trinh/code/ViLa/quilt-llava/playground/data/eval/quiltvqa/answers/$CKPT_name-w-yn-quilt.jsonl \
    --temperature 0
#    --conv-mode vicuna_v1

# Evaluate
# QUILT-VQA
python llava/eval/quilt_eval.py \
    --quilt True \
    --gt /ssd1/trinh/data/ViLa/images/quilt_instruct/Quilt_LLaVA/Quilt_VQA/quiltvqa_test_w_ans.json \
    --pred /data1/trinh/code/ViLa/quilt-llava/playground/data/eval/quiltvqa/answers/$CKPT_name-w-yn-quilt.jsonl


CKPT_name="llava-onevision-siglip-Qwen2.5-7B-Instruct-ov_PathResVL_instruct_SC_video_v1.0_gpt4o_mini_diagnosis_short_yolo_bs2_from_im+clip_lora_merge" # or "./checkpoints/...your model"

#('Metric               Performance\n'
# '-----------------  -------------\n'
# 'exact match score        9.34427\n'
# 'f1 score                25.3818\n'
# 'precision               17.7144\n'
# 'recall                  60.5834\n'
# 'yes/no accuracy         56.5598')



## QUILT-VQA
#python -m llava.eval.model_vqa \
#    --model-path $CKPT \
#    --question-file ./playground/data/eval/quiltvqa/quiltvqa_test_wo_ans.jsonl \
#    --image-folder ./playground/data/eval/quiltvqa/images \
#    --answers-file ./playground/data/eval/quiltvqa/answers/$CKPT-w-yn-quilt.jsonl \
#    --temperature 0 \
#    --conv-mode vicuna_v1
#
## Evaluate
## QUILT-VQA
#python llava/eval/quilt_eval.py \
#    --quilt True \
#    --gt ./playground/data/eval/quiltvqa/quiltvqa_test_w_ans.json \
#    --pred ./playground/data/eval/quiltvqa/answers/$CKPT-w-yn-quilt.jsonl
#
#
#
##python llava.eval.model_vqa \
##    --model-path ./checkpoints/LLaVA-13B-v0 \
##    --question-file \
##    playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
##    --image-folder \
##    /path/to/coco2014_val \
##    --answers-file \
##    /path/to/answer-file-our.jsonl
#
#
#
#CKPT="8b-vila-v1.5-mm-sft" # or "./checkpoints/...your model"
#CKPT_path="/data1/trinh/code/ViLa/image_text/VILA_0820/checkpoints/8b/vila-v1.5-mm-sft" # or "./checkpoints/...your model"
#
#CKPT="8b-vila-v1.5-mm-pretrain" # or "./checkpoints/...your model"
#CKPT_path="/data1/trinh/code/ViLa/image_text/VILA_0820/checkpoints/8b/vila-v1.5-mm-pretrain" # or "./checkpoints/...your model"
#
#CKPT="8b-vila-v1.5-mm-sft-video-resi_chunk0_gpt4omini_cls3_video-bs2" # or "./checkpoints/...your model"
#CKPT_path="/data1/trinh/code/ViLa/image_text/VILA_0820/checkpoints/8b/vila-v1.5-mm-sft-video-resi_chunk0_gpt4omini_cls3_video-bs2/checkpoint-400" # or "./checkpoints/...your model"
#
##mkdir -p /data1/trinh/code/ViLa/quilt-llava/playground/data/eval/quiltvqa/answers
#
#
#python -m llava.eval.model_vqa \
#    --model-path $CKPT_path \
#    --question-file /data1/trinh/code/ViLa/quilt-llava/playground/data/eval/quilt_gpt/quilt_gpt_questions.jsonl \
#    --image-folder /ssd1/trinh/data/ViLa/images/quilt_instruct/Quilt_LLaVA/Quilt_VQA/quilt_vqa/images  \
#    --answers-file /data1/trinh/code/ViLa/quilt-llava/playground/data/eval/quilt_gpt/answers/$CKPT-w-yn-quilt.jsonl \
#    --temperature 0 \
#    --conv-mode llama_3

# Evaluate
# QUILT-VQA
#python llava/eval/quilt_eval.py \
#    --quilt True \
#    --gt /data1/trinh/code/ViLa/quilt-llava/playground/data/eval/quilt_gpt/quilt_gpt_answers.jsonl \
#    --pred /data1/trinh/code/ViLa/quilt-llava/playground/data/eval/quilt_gpt/answers/$CKPT-w-yn-quilt.jsonl


##/ssd1/trinh/data/ViLa/images/quilt_instruct/Quilt_LLaVA/Quilt_VQA/
#
## QUILT-VQA
#python -m llava.eval.model_vqa \
#    --model-path $CKPT \
#    --question-file /ssd1/trinh/data/ViLa/images/quilt_instruct/Quilt_LLaVA/Quilt_VQA/quiltvqa_test_wo_ans.jsonl \
#    --image-folder  /ssd1/trinh/data/ViLa/images/quilt_instruct/Quilt_LLaVA/Quilt_VQA/quilt_vqa/images \
#    --answers-file ./playground/data/eval/quiltvqa/answers/$CKPT-w-yn-quilt.jsonl \
#    --temperature 0 \
#    --conv-mode vicuna_v1
#
## Evaluate
## QUILT-VQA
#python llava/eval/quilt_eval.py \
#    --quilt True \
#    --gt /ssd1/trinh/data/ViLa/images/quilt_instruct/Quilt_LLaVA/Quilt_VQA/quiltvqa_test_w_ans.json \
#    --pred ./playground/data/eval/quiltvqa/answers/data1/trinh/code/ViLa/quilt-llava/checkpoints/quilt-llava-v1.5-7b-f-1eps-w-yn-quilt.jsonl
#


## QUILT-VQA
#python -m llava.eval.model_vqa \
#    --model-path $CKPT \
#    --question-file ./playground/data/eval/quiltvqa/quiltvqa_test_wo_ans.jsonl \
#    --image-folder ./playground/data/eval/quiltvqa/images \
#    --answers-file ./playground/data/eval/quiltvqa/answers/$CKPT-w-yn-quilt.jsonl \
#    --temperature 0 \
#    --conv-mode vicuna_v1
#
## Evaluate
## QUILT-VQA
#python llava/eval/quilt_eval.py \
#    --quilt True \
#    --gt ./playground/data/eval/quiltvqa/quiltvqa_test_w_ans.json \
#    --pred ./playground/data/eval/quiltvqa/answers/$CKPT-w-yn-quilt.jsonl
#


#1283/1283 [50:16<00:00,  2.35s/it]
#/Quilt-Llava-v1.5-7b-w-yn-quilt.jsonl
# ('Metric               Performance\n'
#  '-----------------  -------------\n'
#  'exact match score        15.7358\n'
#  'f1 score                 31.8981\n'
#  'precision                24.5947\n'
#  'recall                   53.8973\n'
#  'yes/no accuracy          62.9738')
#
#
#CKPT="/data1/trinh/code/ViLa/quilt-llava/checkpoints/quilt-llava-v1.5-7b-f-1eps" # or "./checkpoints/...your model"

#('Metric               Performance\n'
 # '-----------------  -------------\n'
 # 'exact match score        12.3638\n'
 # 'f1 score                 29.9131\n'
 # 'precision                21.7772\n'
 # 'recall                   57.5018\n'
 # 'yes/no accuracy          65.0146')

#"context":" Sharing  the  screen.  In  the mode of  thinking  low  power  first,  and  then  high power.  We  are with  a  nice low  power view.  I  think I  got  the  impression  it  was  quite  hard  for  people  to  work  out  what  was  happening  when  they  first  saw  this. [0.43, 0.63, 0.84, 0.69],  Yes.  The  pattern  is  one  that's  really  characteristic,  and  pathologists  look  at  this  and  go,  oh.  OK,  we  go,  oh,  because  what  we  recognise  are  these  little  round,  little  round cannonball  things.  Yes,  but  the tricky [0.0, 0.0, 0.83, 0.66]", "type": "Hematopathology"
#
#"question": "How would you encapsulate the findings presented in the histopathology image?"
#"ground truth": "Observing the histopathology image, I am initially drawn to a region in the upper right quadrant. At a low magnification, it may be challenging to discern the details, but upon closer inspection, a distinct pattern emerges. This pattern is characterized by the presence of numerous small, round structures that resemble cannonballs. These structures are a key feature that pathologists often recognize. Shifting focus to a larger area that encompasses almost the entire image, excluding the upper right and lower right quadrants, the complexity of the image becomes more apparent. This area is somewhat more challenging to interpret due to its intricate nature. The histological features here require a more detailed examination to fully understand their implications."
#
#"pred 1": "Upon examination of the histopathology image, I observe a lymph node with a distinct follicular pattern. The follicles are well-defined, indicating a normal lymphoid architecture. The germinal centers are prominent and appear to be in a healthy state, suggesting active B-cell proliferation.\n\nIn the upper left quadrant of the image, there is a significant presence of sinus histiocytes. These cells, which are part of the monocyte/macrophage lineage, are typically found in the sinuses of the lymph node. They appear to be in a healthy state, with no signs of abnormal proliferation or dysplasia.\n\nMoving towards the center and slightly to the right, there is a region that appears to be a reactive lymph node. This is characterized by an increase in the number of lymphocytes, plasma cells, and macrophages, which is a common response to infection or inflammation. The architecture of this region is preserved, which is typical of a reactive lymph node. The presence of follicles and sinuses further supports this interpretation."
#"pred 2": "Upon examination of the histopathology image, I observe a region of interest that appears to be a lymph node. This area is characterized by a dense aggregation of cells, indicative of a lymphoid tissue. The cells are predominantly small, round, and densely packed, typical of lymphocytes. \n\nIn addition to the lymphoid tissue, there are also areas that resemble normal lymph node architecture. These regions are characterized by the presence of follicles, which are spherical structures composed of densely packed lymphocytes. The follicles are surrounded by a thin layer of connective tissue, known as the mantle zone, which is also visible in the image.\n\nThe overall histological features suggest that this could be a lymph node, although further investigation would be required to confirm this. The presence of both lymphoid tissue and areas resembling normal lymph node architecture could indicate a reactive process or a lymphoproliferative disorder."
#"pred 3": "Upon examining the histopathology image, I observe a well-circumscribed lesion located towards the center and slightly to the right. This lesion is characterized by a distinct, rounded shape, indicative of a benign nature. The lesion is surrounded by a fibrous capsule, which is a common feature of benign tumors. The fibrous capsule appears to be intact and well-defined, further supporting the benign diagnosis. The lesion's well-circumscribed nature and the presence of a fibrous capsule are key features that suggest a benign pathology."
#"pred 4": "Upon examination of the histopathology image, I observe a region of interest that appears to be a benign lesion. This area, located towards the center and slightly to the right, exhibits characteristics of a trichilemmoma. The lesion is well-circumscribed, with a lobulated architecture. The cells within the lesion display clear cytoplasm and prominent, centrally located nuclei, which are typical features of trichilemmoma. \n\nThe surrounding tissue appears normal, with no signs of inflammation or necrosis, which further supports the benign nature of this lesion. The absence of mitotic figures and the lack of cellular atypia further corroborate the benign diagnosis. \n\nIn conclusion, the histopathological features observed in this image are consistent with a diagnosis of trichilemmoma."
#"pred 5": "Upon examination of the histopathology image, I observe a well-defined lymph node. The lymph node is encapsulated by a thin fibrous capsule, which is clearly visible. The capsule is composed of dense fibrous tissue, providing a protective layer for the lymph node.\n\nWithin the lymph node, there are distinct areas of darker staining. These areas are populated by lymphocytes, which are small, round cells with darkly stained nuclei. The lymphocytes are densely packed and uniformly distributed throughout the darker staining areas, indicating a healthy immune response.\n\nThe lymph node is well-preserved, with no signs of necrosis or significant inflammation. The architecture of the lymph node is intact, with clear demarcation between the cortex and medulla. The cortex appears as a lighter stained area with a dense network of lymphocytes, while the medulla is darker and contains fewer lymphocytes.\n\nOverall, the histopathology image provides a detailed view of a well-preserved lymph node, with no signs of pathological changes."
#
