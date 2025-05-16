import argparse
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'


import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re
import requests
import copy
import torch
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

def eval_model(args):

    # Model
    disable_torch_init()
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(args.model_path, None, model_name,
                                                                          device_map=device_map,
                                                                          attn_implementation=None)  # Add any other thing you want to pass in llava_model_args
    model.eval()

    # Data
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file))]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models


    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        question = DEFAULT_IMAGE_TOKEN + f"\n{qs}"

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = process_images([image], image_processor, model.config)
        image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [image.size]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]




        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                do_sample=True if args.temperature > 0 else False,
                image_sizes=image_sizes,
                temperature=args.temperature,
                max_new_tokens=4096,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": qs,
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                }
            )
            + "\n"
        )
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    # model_path = "/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/onevision/llava-onevision-siglip-Qwen2.5-0.5B-Instruct-ov_PathResVL_instruct_video_chunk0_gpt4o_train_cls33+2dms_datav0_video_cc05-anyres-bs2"
    # model_path = "/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/onevision/llava-onevision-siglip-Qwen2.5-0.5B-Instruct-ov_PathResVL_instruct_video_chunk0_gpt4o_train_cls33+2dms_datav0-anyres-bs2"

    model_path = "/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/onevision/llava-onevision-siglip-Qwen2.5-0.5B-Instruct-ov_PathResVL_instruct_video_chunk0_gpt4o_train_cls33_datav0" #2
    model_path = "/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/onevision/llava-onevision-siglip-Qwen2.5-0.5B-Instruct-ov_PathResVL_instruct_video_chunk0_gpt4o_train_cls33_datav0_video_cc05" #1
    model_path = "/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/onevision/llava-onevision-siglip-Qwen2.5-0.5B-Instruct-ov_PathResVL_instruct_video_chunk_yolo_gpt4o_train_cls33+2dms_datav0-anyres-bs2_1box_error" #1
    model_path = "/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/onevision/llava-onevision-siglip-Qwen2.5-0.5B-Instruct-ov_PathResVL_instruct_video_chunk_yolo_gpt4o_train_cls33+2dms_datav0-anyres-bs2_1box_error" #1
    model_path =  "/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/onevision_sc/llava-onevision-siglip-Qwen2.5-7B-Instruct-ov_PathResVL_instruct_SC_video_v0_bs2_from_im+clip_lora_merge"

    model_path = "/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/llavanext-siglip-Qwen2.5-0.5B-Instruct-mlp2x_gelu-pretrain_PathResVL_2dms_instruct"
    # model_path = "/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/onevision/llava-onevision-siglip-Qwen2.5-7B-Instruct-ov_PathResVL_instruct_image_2dms-anyres_bs2"

    # model_path = "/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/onevision_sc_v1.0/llava-onevision-siglip-Qwen2.5-7B-Instruct-ov_PathResVL_instruct_SC_video_v1.0_gpt4o_mini_diagnosis_short_yolo_bs2_from_stage1_im_lora_merge"

    save_name = model_path.split('/')[-1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=model_path)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/ssd1/trinh/data/ViLa/images/quilt_instruct/Quilt_LLaVA/Quilt_VQA/quilt_vqa/images")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--question-file", type=str, default="/data1/trinh/code/ViLa/quilt-llava/playground/data/eval/quilt_gpt/quilt_gpt_questions.jsonl")
    parser.add_argument("--answers-file", type=str, default=f"/data1/trinh/code/ViLa/quilt-llava/playground/data/eval/quilt_gpt/answers/{save_name}-w-yn-quilt.jsonl")
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    args = parser.parse_args()

    eval_model(args)


# /data1/trinh/data/raw_data/quilt1m_data/clips_histo_frame_clean/iHistopathology/HAl5Y4kC1xA/v_HAl5Y4kC1xA_c00022.mp4 too long
