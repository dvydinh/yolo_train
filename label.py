import os
import glob
from ultralytics import YOLO

target_dir = "/content/drive/MyDrive/outpainting_workspace/data/results"
valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

weight_files = glob.glob("runs/smf_fpidet_detection/*/weights/best.pt")
if not weight_files:
    weight_files = glob.glob("runs/smf_fpidet_detection/*/weights/last.pt")

if not weight_files:
    print("model not found")
    exit()

latest_weight = max(weight_files, key=os.path.getmtime)
print("using model:", latest_weight)
model = YOLO(latest_weight)

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
