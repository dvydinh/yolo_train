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

for root, dirs, files in os.walk(target_dir):
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_exts:
            img_path = os.path.join(root, f)
            results = model(img_path, verbose=False)
            
            for res in results:
                txt_path = os.path.splitext(img_path)[0] + ".txt"
                with open(txt_path, "w") as f_txt:
                    for box in res.boxes:
                        c = int(box.cls[0])
                        x, y, w, h = box.xywhn[0].tolist()
                        f_txt.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
