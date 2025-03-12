import os
import requests
import zipfile
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO

# -----------------------------
# ✅ 1. Set Up Dataset Paths
# -----------------------------
BASE_DIR = Path.cwd() / "coco_sample"  # Directory where dataset will be saved
IMAGES_DIR = BASE_DIR / "images"
ANNOTATIONS_DIR = BASE_DIR / "annotations"

# Create directories if they don't exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

# COCO URLs (Instance Segmentation Only)
ANNOTATION_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATION_ZIP = ANNOTATIONS_DIR / "annotations_trainval2017.zip"
ANNOTATION_FILE = ANNOTATIONS_DIR / "annotations" / "instances_val2017.json"
COCO_IMAGE_ZIP_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_IMAGE_ZIP = BASE_DIR / "val2017.zip"

# -----------------------------
# ✅ 2. Download & Extract COCO Annotations
# -----------------------------
def download_annotations():
    """Downloads and extracts COCO annotations if not already present."""
    if not ANNOTATION_FILE.exists():
        print("📥 Downloading COCO annotations...")
        response = requests.get(ANNOTATION_URL, stream=True)
        with open(ANNOTATION_ZIP, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024), desc="Downloading Annotations"):
                f.write(chunk)

        # Extract annotations
        with zipfile.ZipFile(ANNOTATION_ZIP, "r") as zip_ref:
            zip_ref.extractall(ANNOTATIONS_DIR)

        print("✅ Annotations downloaded and extracted.")
    else:
        print("✅ Annotations already exist.")

# -----------------------------
# ✅ 3. Download & Extract COCO Images
# -----------------------------
def download_images():
    """Downloads and extracts COCO validation images."""
    if not IMAGES_DIR.exists() or len(list(IMAGES_DIR.glob("*.jpg"))) == 0:
        print("📥 Downloading COCO validation images...")
        response = requests.get(COCO_IMAGE_ZIP_URL, stream=True)
        with open(COCO_IMAGE_ZIP, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024), desc="Downloading Images"):
                f.write(chunk)

        # Extract images
        with zipfile.ZipFile(COCO_IMAGE_ZIP, "r") as zip_ref:
            zip_ref.extractall(BASE_DIR)

        # Move images to the correct directory
        extracted_img_dir = BASE_DIR / "val2017"
        extracted_img_dir.rename(IMAGES_DIR)

        print("✅ Images downloaded and extracted.")
    else:
        print("✅ Images already exist.")

# -----------------------------
# ✅ 4. Load COCO & Select Sample Images
# -----------------------------
def load_coco_subset(num_images=100):
    """Loads a subset of COCO dataset (default: 100 images)."""
    coco = COCO(str(ANNOTATION_FILE))  # Load COCO annotations
    all_image_ids = coco.getImgIds()
    sample_image_ids = all_image_ids[:num_images]  # Select first 100 images
    print(f"✅ Selected {num_images} images from COCO dataset.")
    return coco, sample_image_ids

# -----------------------------
# ✅ 5. Verify Dataset Integrity
# -----------------------------
def verify_dataset(coco, sample_image_ids):
    """Ensures all selected images exist in the dataset."""
    missing_images = []
    for img_id in sample_image_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = IMAGES_DIR / img_info["file_name"]
        if not img_path.exists():
            missing_images.append(img_info["file_name"])

    if missing_images:
        print(f"❌ Warning: {len(missing_images)} images are missing!")
        print("Missing images:", missing_images[:5])  # Print only first 5
    else:
        print("✅ All images are correctly downloaded.")

# -----------------------------
# ✅ 6. Visualize Sample Images
# -----------------------------
def visualize_images(coco, sample_image_ids):
    """Displays a few sample images with annotations."""
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i, img_id in enumerate(sample_image_ids[:5]):  # Show first 5 images
        img_info = coco.loadImgs(img_id)[0]
        img_path = IMAGES_DIR / img_info["file_name"]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        axes[i].imshow(image)
        axes[i].set_title(f"Image ID: {img_id}")
        axes[i].axis("off")

    plt.show()

# -----------------------------
# ✅ 7. Run All Steps
# -----------------------------
if __name__ == "__main__":
    download_annotations()
    download_images()
    coco, sample_image_ids = load_coco_subset(num_images=100)
    verify_dataset(coco, sample_image_ids)
    visualize_images(coco, sample_image_ids)

    print("\n🎉 COCO 2017 Instance Segmentation dataset is ready in 'coco_sample/' folder!")
