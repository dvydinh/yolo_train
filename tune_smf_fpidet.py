from ultralytics import YOLO
import torch

MODEL_PATH = "yolo11x.pt"
DATA_YAML = "data_smf_fpidet.yaml"
PROJECT = "runs/smf_fpidet_detection"
model_size = "x" if "x" in MODEL_PATH else "l" if "l" in MODEL_PATH else "m" if "m" in MODEL_PATH else "s"
NAME = f"yolo11{model_size}_smf_fpidet_tuning"

IMG_SIZE = 1280
BATCH_SIZE = 0.9 
DEVICE = 0
WORKERS = 16

print("=" * 70)
print("YOLO SMF_FPIDET DETECTION - HYPERPARAMETER TUNING")
print("=" * 70)

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(DEVICE)}")

print(f"Model   : {MODEL_PATH}")
print(f"Dataset : {DATA_YAML}")
print(f"Image   : {IMG_SIZE}")
print(f"Batch   : {BATCH_SIZE}")
print("=" * 70)
print("WARNING: This process will run multiple iterations and may take hours/days.")
print("=" * 70)

model = YOLO(MODEL_PATH)
model.tune(
    data=DATA_YAML,
    epochs=30,
    iterations=30,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=WORKERS,
    project=PROJECT,
    name=NAME,
    optimizer="AdamW",
    plots=False,
    save=False,
    val=False
)

print("\n" + "=" * 70)
print("TUNING FINISHED")
print(f"Check {PROJECT}/{NAME} for the best_hyperparameters.yaml file.")
print("=" * 70)
