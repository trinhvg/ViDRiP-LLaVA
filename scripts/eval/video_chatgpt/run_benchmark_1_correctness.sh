

CKPT_NAME="Video-LLaVA-7B"
model_path="/data1/trinh/code/ViLa/image_text/VILA_0820/checkpoints/8b/vila-v1.5-mm-sft-video-chunk-v0s3-bs2"
cache_dir="./cache_dir"
Video_5_Benchmark="eval/Video_5_Benchmark"
video_dir="/data1/trinh/data/raw_data/quilt1m_data/clips_histo_frame/"
gt_file="/data1/trinh/data/raw_data/quilt1m_data/clips_histo_frame/PathResVL_instruct_video_chunk_val_4s3.json"
output_dir="${Video_5_Benchmark}/${CKPT_NAME}"
output_name="correctness_qa"

python3 llava/eval/video/run_inference_benchmark_general.py \
    --model_path ${model_path} \
    --cache_dir ${cache_dir} \
    --video_dir ${video_dir} \
    --gt_file ${gt_file} \
    --output_dir ${output_dir} \
    --output_name ${output_name}
#
#CKPT_NAME="Video-LLaVA-7B"
#model_path="checkpoints/${CKPT_NAME}"
#cache_dir="./cache_dir"
#Video_5_Benchmark="eval/Video_5_Benchmark"
#video_dir="${Video_5_Benchmark}/Test_Videos"
#gt_file="${Video_5_Benchmark}/Benchmarking_QA/generic_qa.json"
#output_dir="${Video_5_Benchmark}/${CKPT_NAME}"
#output_name="correctness_qa"
#
#python3 llava/eval/video/run_inference_benchmark_general.py \
#    --model_path ${model_path} \
#    --cache_dir ${cache_dir} \
#    --video_dir ${video_dir} \
#    --gt_file ${gt_file} \
#    --output_dir ${output_dir} \
#    --output_name ${output_name}
