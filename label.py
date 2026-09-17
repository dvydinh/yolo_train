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

import datetime

# Lấy tên thư mục của model (ví dụ: yolo26m_fixed_patience20) và nối thêm mốc thời gian để không bao giờ bị trùng
model_run_name = os.path.basename(os.path.dirname(os.path.dirname(best_model_path)))
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
unique_folder_name = f"label_results_{model_run_name}_{timestamp}"

output_base_dir = os.path.join("/content/drive/MyDrive/yolo_train", unique_folder_name)
labels_dir = os.path.join(output_base_dir, "labels")
images_bb_dir = os.path.join(output_base_dir, "images_bb")
print(f"[*] THƯ MỤC LƯU KẾT QUẢ ĐỘC LẬP: {output_base_dir}")

batch_size = 64
img_batch = []

def process_batch(batch):
    if not batch: return
    # Model nhận nguyên list ảnh để xử lý song song (batch processing)
    results = model(batch, verbose=False)
    for res in results:
        original_img_path = res.path
        # Tính toán đường dẫn tương đối để giữ nguyên cấu trúc thư mục (tránh trùng tên file)
        rel_path = os.path.relpath(original_img_path, target_dir)
        rel_dir = os.path.dirname(rel_path)
        filename = os.path.basename(rel_path)
        base_name = os.path.splitext(filename)[0]
        
        out_label_dir = os.path.join(labels_dir, rel_dir)
        out_image_dir = os.path.join(images_bb_dir, rel_dir)
        os.makedirs(out_label_dir, exist_ok=True)
        os.makedirs(out_image_dir, exist_ok=True)
        
        txt_path = os.path.join(out_label_dir, base_name + ".txt")
        img_bb_path = os.path.join(out_image_dir, filename)
        
        with open(txt_path, "w") as f_txt:
            for box in res.boxes:
                c = int(box.cls[0])
                x, y, w, h = box.xywhn[0].tolist()
                f_txt.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                
        # Lưu tấm ảnh đã được model vẽ sẵn bounding box
        res.save(filename=img_bb_path)

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
