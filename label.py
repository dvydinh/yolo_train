import os
import glob
from ultralytics import YOLO

target_dir = "/content/drive/MyDrive/outpainting_workspace/data/results"
valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

import pandas as pd

# Đệ quy tìm tất cả các file best.pt ở mọi ngóc ngách
pt_files = glob.glob("**/*/weights/best.pt", recursive=True)
best_model_path = None

if pt_files:
    # Lấy file best.pt mới được sinh ra gần đây nhất (chính là bản multiscale vừa train xong)
    best_model_path = max(pt_files, key=os.path.getmtime)
    print(f"Auto-selected the newest model: {best_model_path}")

if not best_model_path:
    print("model not found")
    exit()

model = YOLO(best_model_path)

batch_size = 64
img_batch = []

def process_batch(batch):
    if not batch: return
    # Model nhận nguyên list ảnh để xử lý song song (batch processing)
    results = model(batch, verbose=False)
    for res in results:
        # Lấy lại đường dẫn ảnh gốc từ kết quả
        original_img_path = res.path
        txt_path = os.path.splitext(original_img_path)[0] + ".txt"
        with open(txt_path, "w") as f_txt:
            for box in res.boxes:
                c = int(box.cls[0])
                x, y, w, h = box.xywhn[0].tolist()
                f_txt.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

for root, dirs, files in os.walk(target_dir):
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_exts:
            img_path = os.path.join(root, f)
            img_batch.append(img_path)
            
            # Đủ 64 tấm thì tống vào GPU xử lý 1 lần
            if len(img_batch) >= batch_size:
                process_batch(img_batch)
                img_batch = []

# Xử lý nốt những tấm lẻ tẻ còn sót lại cuối cùng
process_batch(img_batch)
print("Labeling hoàn tất!")
