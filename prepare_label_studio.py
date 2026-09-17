import os
import glob
import shutil
from pathlib import Path
import zipfile

print("="*50)
print("CHUẨN BỊ DỮ LIỆU ĐỂ ĐƯA LÊN LABEL STUDIO")
print("="*50)

# 1. Định vị các thư mục
original_images_dir = "/content/drive/MyDrive/outpainting_workspace/data/results"
yolo_train_dir = "/content/drive/MyDrive/yolo_train"

# Tìm thư mục label_results mới nhất
result_dirs = glob.glob(f"{yolo_train_dir}/label_results_*")
if not result_dirs:
    print("Không tìm thấy thư mục label_results nào!")
    exit()

latest_results_dir = max(result_dirs, key=os.path.getmtime)
labels_dir = os.path.join(latest_results_dir, "labels")

print(f"[*] Đang lấy nhãn từ đợt chạy mới nhất: {latest_results_dir}")

# 2. Tạo thư mục tạm để nén
import tempfile
temp_dir = tempfile.mkdtemp()
ls_images_dir = os.path.join(temp_dir, "images")
ls_labels_dir = os.path.join(temp_dir, "labels")
os.makedirs(ls_images_dir, exist_ok=True)
os.makedirs(ls_labels_dir, exist_ok=True)

# 3. Tạo file classes.txt bắt buộc cho Label Studio
classes_file = os.path.join(temp_dir, "classes.txt")
with open(classes_file, "w") as f:
    f.write("person\nphone\n")
print("[*] Đã tạo file classes.txt (0: person, 1: phone)")

# 4. Sao chép nhãn (.txt) và ảnh gốc tương ứng
txt_files = glob.glob(f"{labels_dir}/**/*.txt", recursive=True)
count = 0

valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

for txt_path in txt_files:
    # Lấy đường dẫn tương đối để tìm ảnh gốc
    rel_path = os.path.relpath(txt_path, labels_dir)
    base_name = os.path.splitext(rel_path)[0]
    
    # Tìm file ảnh gốc tương ứng
    img_found = False
    for ext in valid_exts:
        potential_img = os.path.join(original_images_dir, base_name + ext)
        if os.path.exists(potential_img):
            # Copy ảnh GỐC (chưa bị vẽ khung) vào thư mục images
            shutil.copy2(potential_img, os.path.join(ls_images_dir, os.path.basename(potential_img)))
            img_found = True
            break
            
    if img_found:
        # Copy file txt vào thư mục labels
        shutil.copy2(txt_path, os.path.join(ls_labels_dir, os.path.basename(txt_path)))
        count += 1

print(f"[*] Đã sao chép thành công {count} cặp Ảnh + Nhãn (Gốc)")

# 5. Nén lại thành file ZIP
zip_path = os.path.join(yolo_train_dir, "label_studio_import.zip")
print(f"[*] Đang nén thành file {zip_path} ...")

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Thêm classes.txt
    zipf.write(classes_file, arcname="classes.txt")
    # Thêm images
    for root, _, files in os.walk(ls_images_dir):
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, arcname=f"images/{file}")
    # Thêm labels
    for root, _, files in os.walk(ls_labels_dir):
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, arcname=f"labels/{file}")

print("="*50)
print(f"✅ HOÀN TẤT! File nén đã sẵn sàng tại: {zip_path}")
print("👉 Hướng dẫn đưa lên Label Studio:")
print("1. Mở Label Studio -> Create New Project")
print("2. Vào Settings -> Labeling Interface -> Chọn 'Object Detection with Bounding Boxes'")
print("3. Xóa các nhãn cũ, nhập vào 2 nhãn: person, phone")
print("4. Bấm Import -> Cầm file label_studio_import.zip thả vào là xong!")
print("="*50)
