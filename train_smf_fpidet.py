from ultralytics import YOLO
import torch
from pathlib import Path
import os

MODEL_PATH = "yolo26l.pt"

DATA_YAML = "data_smf_fpidet.yaml"

PROJECT = "runs/smf_fpidet_detection"
model_size = "m" if "m" in MODEL_PATH else "l" if "l" in MODEL_PATH else "s"
NAME = f"yolov9{model_size}_smf_fpidet"

EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 0.9
DEVICE = 0
WORKERS = 16

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
import glob
run_dirs = glob.glob(f"{PROJECT}/{NAME}*")
last_pt = None
if run_dirs:
    latest_run = max(run_dirs, key=os.path.getmtime)
    potential_last = Path(latest_run) / "weights" / "last.pt"
    if potential_last.exists():
        last_pt = potential_last

if last_pt:
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
