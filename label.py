import os
import glob
from ultralytics import YOLO

target_dir = "/content/drive/MyDrive/outpainting_workspace/data/results"
valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

import pandas as pd

runs = glob.glob("runs/smf_fpidet_detection/*")
best_model_path = None
best_map = -1.0

for run in runs:
    csv_file = os.path.join(run, "results.csv")
    pt_file = os.path.join(run, "weights", "best.pt")
    if os.path.exists(csv_file) and os.path.exists(pt_file):
        try:
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
            map_col = [c for c in df.columns if "mAP50-95" in c]
            if map_col:
                max_map = df[map_col[0]].max()
                if max_map > best_map:
                    best_map = max_map
                    best_model_path = pt_file
        except Exception:
            pass

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
