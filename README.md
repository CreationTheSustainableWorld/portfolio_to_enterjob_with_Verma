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

- **Base model**: [`microsoft/git-base`](https://huggingface.co/microsoft/git-base)  
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

## 🔍 Visual Comparison (GitHub Pages)

👉 [Click here to view the caption comparison results online](https://creationthesustainableworld.github.io/portfolio_to_enterjob_with_Verma/)  
*You can visually verify how the model improves after fine-tuning.*

![image](./assets/sample_comparison.png) <!-- ← Optionally replace with actual screenshot -->

---

## 🛠 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python a.py

# 3. Optionally regenerate HTML comparison
python generate_html.py
