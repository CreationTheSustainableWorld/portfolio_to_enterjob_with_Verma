import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AdamW
import evaluate
from datasets import load_dataset

# === 設定 ===
DATA_ORIGINAL_JSON = "dataset/flickr8k_original_raw.json"
DATA_EVAL_JSON = "dataset/flickr8k_eval_formatted.json"
DATA_TRAIN_JSON = "dataset/flickr8k_train_formatted.json"
IMAGE_DIR = "dataset/images"
PRETRAINED_DIR = "models/git_pretrained"
FINETUNED_DIR = "models/git_finetuned_lora"
BEFORE_OUTPUT = "captions_output/generated_before.json"
AFTER_OUTPUT = "captions_output/generated_after.json"
SCORE_OUTPUT = "captions_output/bleu_score_comparison.json"
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-4
MAX_LENGTH = 32
NUM_EVAL_SAMPLES = 20

os.makedirs("dataset", exist_ok=True)
os.makedirs("captions_output", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(PRETRAINED_DIR, exist_ok=True)
os.makedirs(FINETUNED_DIR, exist_ok=True)

# === データ整形（1000件抽出＋画像保存） ===
print("\U0001F4E6 Flickr8k データセットの読み込みと整形中...")
dataset = load_dataset("tsystems/flickr8k", split="train").shuffle(seed=42).select(range(1000))
raw_data = []
eval_data = []
train_data = []

for item in dataset:
    image = item["image"]
    filename = f"{IMAGE_DIR}/{item['image_filename']}"
    try:
        image.save(filename)
    except Exception as e:
        print(f"❌ 画像保存に失敗: {filename} ({e})")
        continue

    entry = {
        "image_filename": item["image_filename"],
        "image_path": filename,
        "captions": item["captions"]
    }
    raw_data.append(entry)
    eval_data.append({
        "image": filename,
        "captions": item["captions"]
    })
    train_data.append({
        "image": filename,
        "caption": item["captions"][0]
    })

with open(DATA_ORIGINAL_JSON, "w", encoding="utf-8") as f:
    json.dump(raw_data, f, ensure_ascii=False, indent=2)
with open(DATA_EVAL_JSON, "w", encoding="utf-8") as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=2)
with open(DATA_TRAIN_JSON, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

print(f"✅ オリジナルデータ: {len(raw_data)} 件 → 学習用: {len(train_data)} 件, 出力用: {len(eval_data)} 件")

# === モデル・プロセッサ準備 ===
print("🔄 モデル・プロセッサの読み込み...")
processor = AutoProcessor.from_pretrained("microsoft/git-base")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

if not os.listdir(PRETRAINED_DIR):
    model.save_pretrained(PRETRAINED_DIR)
    processor.save_pretrained(PRETRAINED_DIR)
    print(f"✅ Pretrained model saved to {PRETRAINED_DIR}")

# === BLEU準備 ===
bleu = evaluate.load("bleu")

# === 推論前キャプション生成 ===
def generate_and_save(model, processor, data, output_path, tag="before"):
    model.eval()
    results = []
    predictions = []
    references = []
    for idx, item in enumerate(tqdm(data[:NUM_EVAL_SAMPLES], desc=f"Generating {tag}")):
        image = Image.open(item["image"]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=MAX_LENGTH)
        caption = processor.tokenizer.decode(out[0], skip_special_tokens=True)
        results.append({
            "id": idx,
            "image_filename": item["image"],
            "generated_caption": caption,
            "reference_captions": item["captions"]
        })
        predictions.append(caption)
        references.append(item["captions"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return predictions, references

with open(DATA_EVAL_JSON, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

pred_before, ref_before = generate_and_save(model, processor, eval_data, BEFORE_OUTPUT, tag="before")

# === LoRAファインチューニング ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

with open(DATA_TRAIN_JSON, "r", encoding="utf-8") as f:
    train_data = json.load(f)

class FlickrDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image"]).convert("RGB")
        caption = item["caption"]

        visual_inputs = self.processor(images=image, return_tensors="pt")
        labels = self.processor.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True
        )

        return {
            "pixel_values": visual_inputs["pixel_values"].squeeze(0),
            "input_ids": labels["input_ids"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0),
            "attention_mask": labels["attention_mask"].squeeze(0)
        }

print("🚀 LoRAファインチューニング開始...")
dataset = FlickrDataset(train_data, processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"✅ Epoch {epoch+1} 完了 - Loss: {total_loss / len(dataloader):.4f}")

print("💾 学習済みモデル保存中...")
model.save_pretrained(FINETUNED_DIR)
processor.save_pretrained(FINETUNED_DIR)

# === 学習後キャプション生成 ===
pred_after, ref_after = generate_and_save(model, processor, eval_data, AFTER_OUTPUT, tag="after")

# === BLEUスコア比較 ===
bleu_before = bleu.compute(predictions=pred_before, references=ref_before)
bleu_after = bleu.compute(predictions=pred_after, references=ref_after)

summary = {
    "bleu_before": bleu_before["bleu"],
    "bleu_after": bleu_after["bleu"]
}

with open(SCORE_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n🎯 スコア比較完了:")
print(json.dumps(summary, indent=2))