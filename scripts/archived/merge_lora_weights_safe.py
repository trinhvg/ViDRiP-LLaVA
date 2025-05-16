import argparse
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def lxr_load_llava_next_ov(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", attn_implementation="flash_attention_2", customized_config=None, overwrite_config=None, **kwargs):
    kwargs["device_map"] = device_map
    kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, **kwargs)

    rank0_print(f"Model Class: {model.__class__.__name__}")
    image_processor = None

    if "llava" in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != "auto":
            vision_tower.to(device="cuda", dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len

def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)

    # Load model (device_map="cpu" can also be "auto" if you have GPUs)
    tokenizer, model, image_processor, context_len = lxr_load_pretrained_model(args.model_path, args.model_base, model_name)

    model.save_pretrained(args.save_model_path, safe_serialization=True)
    tokenizer.save_pretrained(args.save_model_path)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=False, default="/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/onevision_sc/llava-onevision-siglip-Qwen2.5-7B-Instruct-ov_PathResVL_instruct_SC_video_v0_bs2_from_im+clip_lora")
    parser.add_argument("--model-base", type=str, required=False, default="/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/onevision/llava-onevision-siglip-Qwen2.5-7B-Instruct-ov_PathResVL_instruct_video_chunk_yolo_gpt4o_train_cls33_all+2dms-anyres_bs2")
    parser.add_argument("--save-model-path", type=str, required=False, default="/data1/trinh/code/ViLa/video/LLaVA-NeXT/checkpoints/onevision_sc/llava-onevision-siglip-Qwen2.5-7B-Instruct-ov_PathResVL_instruct_SC_video_v0_bs2_from_im+clip_lora_merge")

    args = parser.parse_args()
    merge_lora(args)
