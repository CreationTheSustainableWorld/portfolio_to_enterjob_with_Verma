import os
import json
import base64
from html import escape
from PIL import Image
from io import BytesIO

BEFORE_JSON = "captions_output/generated_before.json"
AFTER_JSON = "captions_output/generated_after.json"
OUTPUT_HTML = "captions_output/comparison_embedded.html"
NUM_SAMPLES = 20  # 表示したい画像数

os.makedirs("captions_output", exist_ok=True)

with open(BEFORE_JSON, "r", encoding="utf-8") as f:
    before_data = json.load(f)
with open(AFTER_JSON, "r", encoding="utf-8") as f:
    after_data = json.load(f)

rows = []
for b, a in zip(before_data[:NUM_SAMPLES], after_data[:NUM_SAMPLES]):
    image_path = b["image_filename"]

    # base64画像に変換
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
            img_tag = f'<img src="data:image/jpeg;base64,{encoded}" width="200">'
    except Exception as e:
        img_tag = f"<i>画像読み込みエラー: {e}</i>"

    row = f"""
    <tr>
        <td>{img_tag}</td>
        <td>{escape(b["generated_caption"])}</td>
        <td>{escape(a["generated_caption"])}</td>
        <td>{'<br>'.join([escape(ref) for ref in b['reference_captions']])}</td>
    </tr>
    """
    rows.append(row)

html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Fine-tuning Comparison</title>
    <style>
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; vertical-align: top; }}
        th {{ background-color: #f0f0f0; }}
        img {{ max-height: 150px; }}
    </style>
</head>
<body>
    <h2>📸 Caption Comparison: Before vs After Fine-Tuning</h2>
    <table>
        <tr>
            <th>Image</th>
            <th>Before (Pretrained)</th>
            <th>After (Fine-tuned)</th>
            <th>Reference Captions</th>
        </tr>
        {''.join(rows)}
    </table>
</body>
</html>
"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ HTML出力完了: {OUTPUT_HTML}")
