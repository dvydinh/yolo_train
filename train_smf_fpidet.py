from ultralytics import YOLO
import torch
from pathlib import Path

MODEL_PATH = "yolo26m.pt" #
# MODEL_PATH = "yolo26l.pt" #
# MODEL_PATH = "yolov9s.pt" #

DATA_YAML = "data_smf_fpidet.yaml"

PROJECT = "runs/smf_fpidet_detection"
NAME = "yolov9_smf_fpidet" #

EPOCHS = 100 #
IMG_SIZE = 640 #
BATCH_SIZE = -1 # AutoBatch: tự động tìm batch_size to nhất vừa với 94GB VRAM
DEVICE = 0
WORKERS = 16 # Tăng luồng nạp data

print("=" * 70)
print("YOLO SMF_FPIDET DETECTION - TRAINING")
print("=" * 70)

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(DEVICE)}")

print(f"Model   : {MODEL_PATH}")
print(f"Dataset : {DATA_YAML}")
print(f"Epochs  : {EPOCHS}")
print(f"Image   : {IMG_SIZE}")
print(f"Batch   : {BATCH_SIZE}")
print("=" * 70)

last_pt = Path(PROJECT) / NAME / "weights" / "last.pt"
if last_pt.exists():
    print(f"Resuming training from {last_pt}")
    model = YOLO(last_pt)
    results = model.train(resume=True)
else:
    model = YOLO(MODEL_PATH)
    results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=WORKERS,
    project=PROJECT,
    name=NAME,
    save=True,
    save_period=10,
    val=True,
    cache=True,
    amp=True,
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    seed=42,
    deterministic=True,
    verbose=True,
)

best_model_path = Path(PROJECT) / NAME / "weights" / "best.pt"

print("\n" + "=" * 70)
print("TRAINING FINISHED")
print("=" * 70)

if best_model_path.exists():
    print(f"Best model: {best_model_path}")
    print("\n" + "=" * 70)
    print("RUNNING FINAL EVALUATION ON TEST SET...")
    print("=" * 70)
    
    best_model = YOLO(best_model_path)
    
    try:
        best_model.val(data=DATA_YAML, split="test")
    except Exception as e:
        print("Evaluation failed:", e)
else:
    print("WARNING: best.pt was not found.")

print("=" * 70)
