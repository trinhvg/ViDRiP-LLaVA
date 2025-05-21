import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pretrained = "./VidDiag_LLaVA_image"
pretrained = "./VidDiag_LLaVA_video"

img_path = ""
video_path = "./example_video/d0WDjz9JBiU_c00003.mp4"

video = True

model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa")

model.eval()


# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()

    # frame_time = [i/vr.get_avg_fps() for i in frame_idx]

    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)




if video:
    video_frames = load_video(video_path, 32)

    print(video_frames.shape) # (16, 1024, 576, 3)
    image_tensors = []
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
    image_tensors.append(frames)
    image_sizes = [frame.size for frame in video_frames]

else:


    image = Image.open(img_path)

    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
    image_sizes = [image.size]
    image_tensors = image_tensor


# Prepare conversation input
conv_template = "qwen_1_5"

question = f"{DEFAULT_IMAGE_TOKEN}\nWhat is the best diagnosis for this ovary/fallopian tube tissue? First, describe the image information relevant to the question. Then, provide your answer."

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

prompt = prompt_question

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)


# Generate response
cont = model.generate(
    input_ids,
    images=image_tensors,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
    modalities=["video"],
    # return_dict_in_generate=True,
    # output_attentions=True,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs[0])

