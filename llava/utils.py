import datetime
import logging
import logging.handlers
import os
import sys
import numpy as np
import cv2
import requests

from llava.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "I am sorry. Your input may violate our content moderation guidelines. Please avoid using harmful or offensive content."

handler = None

import torch.distributed as dist

try:
    import av
    from decord import VideoReader, cpu
except ImportError:
    print("Please install pyav to use video processing functions.")

def process_video_with_decord(video_file, data_args):
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    avg_fps = round(vr.get_avg_fps() / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    frame_time = [i/avg_fps for i in frame_idx]


    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound or data_args.force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i/vr.get_avg_fps() for i in frame_idx]

    video = vr.get_batch(frame_idx).asnumpy()
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    num_frames_to_sample = num_frames = len(frame_idx)
    # https://github.com/dmlc/decord/issues/208
    vr.seek(0)
    return video, video_time, frame_time, num_frames_to_sample


def center_crop_and_resize(image, crop_x=0.5, crop_y=0.5):
    # Get the original image dimensions
    original_height, original_width, _ = image.shape

    # Calculate the new dimensions for cropping
    new_width = int(original_width * crop_x)
    new_height = int(original_height * crop_y)

    # Find the center of the image
    center_x = original_width // 2
    center_y = original_height // 2

    # Calculate the cropping bounds
    start_x = center_x - (new_width // 2)
    end_x = center_x + (new_width // 2)
    start_y = center_y - (new_height // 2)
    end_y = center_y + (new_height // 2)

    # Perform the crop
    cropped_image = image[start_y:end_y, start_x:end_x]

    # Resize the cropped image back to the original size
    resized_image = cv2.resize(cropped_image, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

def process_video_with_decord_center(video_file, data_args):
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    avg_fps = round(vr.get_avg_fps() / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    # frame_time = [i/avg_fps for i in frame_idx]

    crop_ratio = data_args.video_center_crop
    max_frames_num_half = data_args.frames_upbound//2

    if max_frames_num_half > 0:
        if len(frame_idx) > max_frames_num_half or data_args.force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num_half, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            # frame_time = [i/vr.get_avg_fps() for i in frame_idx]


    frame_idx = [item for item in frame_idx for _ in range(2)]
    video = vr.get_batch(frame_idx).asnumpy()

    # Apply center_crop_and_resize only to frames where idx//2 == 1
    video = [center_crop_and_resize(video[idx], crop_ratio, crop_ratio) if (idx % 2 == 1) else video[idx] for idx in range(len(video))]

    # frame_idx = [item for item in frame_idx for _ in range(2)]

    frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    num_frames_to_sample = num_frames = len(frame_idx)
    # https://github.com/dmlc/decord/issues/208
    vr.seek(0)
    return video, video_time, frame_time, num_frames_to_sample

def process_video_with_pyav(video_file, data_args):
    container = av.open(video_file)
    # !!! This is the only difference. Using auto threading
    container.streams.video[0].thread_type = "AUTO"

    video_frames = []
    for packet in container.demux():
        if packet.stream.type == 'video':
            for frame in packet.decode():
                video_frames.append(frame)
    total_frame_num = len(video_frames)
    video_time = video_frames[-1].time
    avg_fps = round(total_frame_num / video_time / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]

    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()


    frames = [video_frames[i] for i in frame_idx]
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_print(*args):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(filename, when="D", utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False
    except KeyError as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"
