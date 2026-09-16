import os
import random
from pathlib import Path
from PIL import Image, ImageDraw

IMAGES_DIR = Path("smf_fpidet_dataset/train/images")
LABELS_DIR = Path("smf_fpidet_dataset/train/labels")
OUTPUT_DIR = Path("samples_to_check")

CLASSES = {0: "person", 1: "phone"}
COLORS = {0: "green", 1: "red"}

def check_labels():
    if not IMAGES_DIR.exists():
        print(f"Error: Directory {IMAGES_DIR} not found.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    
    all_images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        all_images.extend(IMAGES_DIR.glob(ext))
    
    if not all_images:
        print(f"Error: No images found in {IMAGES_DIR}.")
        return

    print(f"Found {len(all_images)} images. Processing 5 random samples...")
    sample_images = random.sample(all_images, min(5, len(all_images)))
    
    for i, img_path in enumerate(sample_images):
        txt_path = LABELS_DIR / (img_path.stem + ".txt")
        
        try:
            img = Image.open(str(img_path)).convert("RGB")
        except Exception as e:
            print(f"Error: Could not read image {img_path} - Reason: {e}")
            continue
            
        w, h = img.size
        draw = ImageDraw.Draw(img)
        
        if txt_path.exists():
            with open(txt_path, "r") as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center, y_center, box_w, box_h = map(float, parts[1:5])
                    
                    x1 = (x_center - box_w/2) * w
                    y1 = (y_center - box_h/2) * h
                    x2 = (x_center + box_w/2) * w
                    y2 = (y_center + box_h/2) * h
                    
                    color = COLORS.get(cls_id, "white")
                    label = CLASSES.get(cls_id, str(cls_id))
                    
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    draw.text((x1, max(0, y1 - 15)), label, fill=color)
        else:
            print(f"Warning: No label file found for {img_path.name}")
            
        out_file = OUTPUT_DIR / f"sample_{i+1}_{img_path.name}"
        img.save(str(out_file))
        print(f"Saved: {out_file}")

if __name__ == "__main__":
    check_labels()
