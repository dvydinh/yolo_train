import os
from pathlib import Path

DATASET_DIR = Path("smf_fpidet_dataset")

def clean_dataset():
    if not DATASET_DIR.exists():
        print(f"Error: Directory not found {DATASET_DIR}")
        return

    classes_files = []
    if (DATASET_DIR / "classes.txt").exists():
        classes_files.append(DATASET_DIR / "classes.txt")
    for split in ["train", "valid", "test"]:
        split_class_file = DATASET_DIR / split / "classes.txt"
        if split_class_file.exists():
            classes_files.append(split_class_file)

    if not classes_files:
        print(f"Error: File classes.txt not found anywhere in {DATASET_DIR}")
        return

    with open(classes_files[0], "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    print("Classes in Label Studio:", classes)

    try:
        person_id = classes.index("person")
        phone_id = classes.index("phone")
    except ValueError as e:
        print(f"Error finding classes: {e}")
        return

    delete_id = classes.index("zDelete") if "zDelete" in classes else -1
    valid_id = classes.index("zValidated") if "zValidated" in classes else -1

    new_mapping = {
        person_id: 0,
        phone_id: 1
    }

    splits = ["train", "valid", "test"]
    deleted_count = 0
    ignored_unvalidated_count = 0
    cleaned_count = 0
    orphaned_images_count = 0

    for split in splits:
        labels_dir = DATASET_DIR / split / "labels"
        images_dir = DATASET_DIR / split / "images"

        if not labels_dir.exists() or not images_dir.exists():
            continue

        for img_path in images_dir.iterdir():
            if not img_path.is_file() or img_path.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
                continue

            txt_path = labels_dir / (img_path.stem + ".txt")

            if not txt_path.exists():
                img_path.unlink()
                orphaned_images_count += 1
                continue

            with open(txt_path, "r") as f:
                lines = f.readlines()

            should_delete_image = False
            has_validated_label = False
            new_lines = []

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                
                class_id = int(parts[0])

                if class_id == delete_id:
                    should_delete_image = True
                    break
                elif class_id == valid_id:
                    has_validated_label = True
                    continue
                elif class_id in new_mapping:
                    parts[0] = str(new_mapping[class_id])
                    new_lines.append(" ".join(parts) + "\n")

            if should_delete_image or not has_validated_label:
                img_path.unlink()
                txt_path.unlink()
                
                if should_delete_image:
                    deleted_count += 1
                else:
                    ignored_unvalidated_count += 1
            else:
                with open(txt_path, "w") as f:
                    f.writelines(new_lines)
                cleaned_count += 1

    for cf in classes_files:
        with open(cf, "w", encoding="utf-8") as f:
            f.write("person\nphone\n")

    print("\n" + "="*50)
    print("CLEANUP FINISHED!")
    print(f"- Deleted {orphaned_images_count} orphaned images (no .txt label).")
    print(f"- Deleted {deleted_count} images (marked as zDelete).")
    print(f"- Ignored {ignored_unvalidated_count} images (not marked as zValidated yet).")
    print(f"- Cleaned and kept {cleaned_count} VALIDATED images for training.")
    print("="*50)

if __name__ == "__main__":
    clean_dataset()
