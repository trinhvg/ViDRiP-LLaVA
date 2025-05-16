#!/bin/bash
set -e


#model_path="/data1/trinh/code/ViLa/image_text/VILA_0820/checkpoints/8b/vila-v1.5-mm-sft-video-chunk-v0s3-bs2" # first argument is the model path
model_path="/data1/trinh/code/ViLa/image_text/VILA_0820/checkpoints/8b/vila-v1.5-mm-sft-video-chunk-v0_video+image_instruct_2domains-bs4" # first argument is the model path #viim
ckpt_name="Video-LLaVA-7B_all_viv0_im_v0" # second argument is the evaluation output directory name
conv_mode=llama_3
result_dir=runs/eval/${ckpt_name}/videochatgpt
if [ "$#" -ge 3 ]; then
    conv_mode="$3"
fi


# general
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

#BENCHMARK=/home/shijial/workspace/LITA-1.5/data/evaluation/video_chatgpt/benchmarking
#VIDEO_DIR="/data1/trinh/data/raw_data/quilt1m_data/clips_histo_frame/"
VIDEO_DIR="/data1/trinh/code/ViLa/image_text/VILA_0820/val_data/videos"

function model_videochatgpt_benchmark {

    gt_file=${1}
    output_dir=${2}
    echo "running ${gt_file}, output to ${output_dir}"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llava/eval/model_videochatgpt_benchmark.py \
            --model-path ${model_path} \
            --image-folder ${VIDEO_DIR} \
            --gt_file ${gt_file} \
            --output_dir ${output_dir} \
            --output_name ${CHUNKS}_${IDX} \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --conv-mode $conv_mode &
    done

    wait

    output_file=${output_dir}/merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${output_dir}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done
}

# general
gt_file="/data1/trinh/code/ViLa/image_text/VILA_0820/val_data/PathResVL_instruct_video_chunk_val_4s3.json"
output_dir="${result_dir}/generic_qa"
model_videochatgpt_benchmark ${gt_file} ${output_dir}

## temporal
#gt_file="${BENCHMARK}/Benchmarking_QA/temporal_qa.json"
#output_dir="${result_dir}/temporal_qa"
#model_videochatgpt_benchmark ${gt_file} ${output_dir}
#
## consistency
#gt_file="${BENCHMARK}/Benchmarking_QA/consistency_qa.json"
#output_dir="${result_dir}/consistency_qa"
#model_videochatgpt_benchmark ${gt_file} ${output_dir}

echo "Model path: $model_path"
echo "Video directory: $VIDEO_DIR"
echo "Ground truth file: $gt_file"
echo "Output directory: $output_dir"

