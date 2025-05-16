
# 🧬 ViDRiP-LLaVA: Multimodal Diagnostic Reasoning in Pathology

**ViDRiP-LLaVA** is a vision-language framework designed for instruction-based diagnostic reasoning using both image patches and video clips from pathology slides. It builds on LLaVA and extends it to the medical domain with domain-specific datasets and fine-tuned models.


🧠 Introducing our ViDRiP-LLaVA: the first multimodal model for diagnostic reasoning in pathology through video-based instruction. 🔬📽️

Our method leverages chain-of-thought (CoT) prompting to distill the reasoning capabilities of LLMs. ViDRiP-LLaVA generates both detailed histological descriptions and final diagnoses, simulating how pathologists analyze and sign out cases.

📚 Trained on 4,278 instructional video pairs

⚙️ Combines single-image + clip transfer and fine-tuning on segmented diagnostic videos


---
<p align="center" width="100%">
<img src="assets/Network.png"  width="80%" height="80%">
</p>


## 📚 Datasets

### 🔹 [ViDRiP_Instruct_Train](https://huggingface.co/datasets/trinhvg/ViDRiP_Instruct_Train)
The videos data is 100 GB, if GoogleDrive doesn't work let use Hugging Face links bellow:
### 🔹 [ViDRiP_Instruct_Train_Video_GoogleDrive](https://drive.google.com/drive/folders/1oxZlaJpE7PGDYt32LeoGgIzwEvWdnupY?usp=sharing)
### 🔹 [ViDRiP_Instruct_Train_Video_Hugging Face](https://huggingface.co/datasets/trinhvg/ViDRiP_Instruct_Train) (There is 10 zip files)

- 4,000+ instruction-style samples
- Each sample includes:
  - A pathology video clip
  - A diagnostic question
  - A multi-turn reasoning answer
- Format: JSON + MP4
- Croissant-compliant metadata for structured use

### 🔹 [ViDRiP_Instruct_Test](https://huggingface.co/datasets/trinhvg/ViDRiP_Instruct_Test)
### 🔹 [ViDRiP_Instruct_Test_Video](https://drive.google.com/drive/folders/1oxZlaJpE7PGDYt32LeoGgIzwEvWdnupY?usp=sharing)

- Held-out test set of diagnostic Q&A pairs
- Used for benchmarking reasoning performance



---

## 🤖 Models

### 🔸 [ViDRiP_LLaVA_video](https://huggingface.co/trinhvg/ViDRiP_LLaVA_video)

- Vision-language model for video-based diagnostic reasoning
- Trained on `ViDRiP_Instruct_Train`
- Suitable for:
  - Medical VQA
  - Instructional explanation generation
  - Educational pathology summarization

### 🔸 [ViDRiP_LLaVA_image](https://huggingface.co/trinhvg/ViDRiP_LLaVA_image)

- Vision-language model for patch-based diagnostic prompts
- Useful for pathology captioning and single-frame inference




## 🚀 Quickstart

### 🔧 Fine-tuning the model on video dataset
```bash
./scripts/train/finetune_ov_video.sh
```

### 🪄 Fine-tuning with LoRA
```bash
./scripts/train/finetune_ov_video_lora.sh
```
🔗 Merge LoRA weights
```bash
./scripts/train/merge_lora_weights.py
```
### 🧪 Usage / Demo
```bash
./doc/ViDRiP_LLaVA_trial.py
```


### 🔧 Evaluate on our video dataset

We use [lmms_eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) to evaluate the performance of video diagnostic reasoning.

To benchmark `ViDRiP-LLaVA` and compare it with other models:

1. Clone the `lmms_eval` repo
2. Copy our evaluation task folder into it:

```bash
cp -r lmms_eval/tasks/ViDRiP_Instruct_Test /path/to/lmms_eval/tasks/
```
You can then run evaluation using the standard lmms_eval CLI interface.

license: cc-by-nc-nd-3.0

### Citation:
Coming soon
