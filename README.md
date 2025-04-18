# 📸 Fine-Tuning Image Captioning with BLIP/GIT on Flickr8k

This project demonstrates how fine-tuning a pre-trained image captioning model using the Flickr8k dataset significantly improves caption quality.  
You can visually compare the outputs **before and after training** and evaluate them with BLEU scores.

---

## ✅ Objective

Show that **LoRA-based fine-tuning** improves the quality of image-to-text captioning.  
This project is part of my AI engineering portfolio.

---

## 📦 Dataset

- **Name**: `tsystems/flickr8k` from Hugging Face
- **Description**: 8,000 images, each with 5 human-written captions
- **Used Subset**: 1,000 images for training & evaluation

---

## 🤖 Model

- **Base model**: `microsoft/git-base`  
- **Fine-tuning**: Performed using [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685)  
- **Frameworks**: 🤗 Transformers, PyTorch, PEFT

---

## 🔁 Training Setup

- **Epochs**: 3  
- **Batch Size**: 4  
- **Learning Rate**: 1e-4  
- **Max Token Length**: 32  
- **LoRA Settings**: r=8, α=32, dropout=0.1

---

## 📊 Evaluation

| Metric | Before Fine-tuning | After Fine-tuning |
|--------|--------------------|-------------------|
| BLEU   | 0.106              | 0.292             |

---

## 🔍 Visual Comparison

You can check the side-by-side caption results in this HTML:
👉 `captions_output/comparison_embedded.html`

![image](./assets/sample_comparison.png) ←（スクショなど追加可能）

---

## 🛠 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all steps in one go
python a.py

# 3. Optionally, regenerate comparison HTML
python generate_html.py
