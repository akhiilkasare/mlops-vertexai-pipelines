import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
#from pycocotools.cocoeval import COCOeval  # Uncomment if you want to evaluate

# ---------------------------
# Configuration - update these paths accordingly.
# ---------------------------
annotations_file = "/Users/akhilarvindkasare/Downloads/segmentation-model/scripts/coco_sample/annotations/annotations/instances_val2017.json"  # e.g., "/Users/yourname/Downloads/COCO/annotations/instances_val2017.json"
images_dir = "/Users/akhilarvindkasare/Downloads/segmentation-model/scripts/coco_sample/images"  # e.g., "/Users/yourname/Downloads/COCO/val2017"
subset_size = 200  # number of images to process for testing
mask_threshold = 0.5  # Threshold for binarizing masks

# ---------------------------
# Load COCO Ground Truth and Build Category Mapping
# ---------------------------
coco_gt = COCO(annotations_file)
# Ensure the ground truth has the 'categories' key.
if "categories" not in coco_gt.dataset:
    raise ValueError("The annotation file does not contain 'categories' information.")

# Build a dictionary mapping COCO category id to category name.
coco_cat_map = {cat["id"]: cat["name"] for cat in coco_gt.dataset["categories"]}
print("COCO Ground Truth Categories:")
print(coco_cat_map)

# ---------------------------
# Create mapping from model's predicted label IDs to COCO category IDs.
# This function matches category names from the model's id2label with the COCO categories.
def create_model_to_coco_mapping(model, coco_gt):
    model_id2label = model.config.id2label  # e.g., {0: "person", 1: "bicycle", ...}
    # Build mapping from COCO category names (lowercase) to id.
    gt_categories = coco_gt.dataset["categories"]
    gt_name_to_id = {cat["name"].lower(): cat["id"] for cat in gt_categories}
    
    mapping = {}
    for model_label, model_cat_name in model_id2label.items():
        model_cat_name_lower = model_cat_name.lower()
        gt_cat_id = gt_name_to_id.get(model_cat_name_lower, None)
        if gt_cat_id is not None:
            mapping[int(model_label)] = int(gt_cat_id)
        else:
            print(f"Warning: Model label {model_label} with name '{model_cat_name}' not found in COCO ground truth.")
    return mapping

# ---------------------------
# Load the pretrained Mask2Former model and its processor.
# ---------------------------
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
model.eval()

# Create the mapping from model label to COCO category
model_to_coco = create_model_to_coco_mapping(model, coco_gt)
print("Model to COCO Mapping:")
print(model_to_coco)

# ---------------------------
# Define a function to visualize predictions.
# ---------------------------
def visualize_prediction(image, instance_map, segments_info, coco_cat_map, model_to_coco, title=""):
    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    overlay = np.zeros((instance_map.shape[0], instance_map.shape[1], 3), dtype=np.uint8)
    for seg in segments_info:
        inst_id = seg["id"]
        score = seg["score"]
        model_label = seg.get("label_id", None)
        if model_label is None:
            continue
        coco_label = model_to_coco.get(int(model_label), None)
        if coco_label is None:
            continue
        label_name = coco_cat_map.get(coco_label, str(coco_label))
        binary_mask = (instance_map == inst_id).astype(np.uint8)
        if np.sum(binary_mask) == 0:
            continue
        color = np.random.randint(0, 255, size=3)
        overlay[binary_mask == 1] = color
        ys, xs = np.where(binary_mask == 1)
        if len(xs) > 0 and len(ys) > 0:
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            plt.text(cx, cy, f"{label_name}: {score:.2f}", fontsize=10, color="white",
                     bbox=dict(facecolor="red", alpha=0.5))
    plt.imshow(overlay, alpha=0.5)
    plt.title(title)
    plt.axis("off")
    plt.show()

# ---------------------------
# Process a subset of images from the COCO validation set.
# ---------------------------
all_image_ids = list(coco_gt.imgs.keys())
selected_ids = all_image_ids[:subset_size]
print(f"Processing {len(selected_ids)} images for prediction...")

predictions = []

for img_id in selected_ids:
    img_info = coco_gt.loadImgs(img_id)[0]
    file_name = img_info["file_name"]
    img_path = os.path.join(images_dir, file_name)
    
    print(f"\nProcessing image {img_id} - {file_name}")
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {file_name}: {e}")
        continue
    
    # Preprocess the image.
    inputs = processor(images=image, return_tensors="pt")
    
    # Run inference.
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Postprocess outputs to get instance segmentation.
    target_size = image.size[::-1]  # (height, width)
    print("Target size:", target_size)
    result = processor.post_process_instance_segmentation(outputs, target_sizes=[target_size])[0]
    
    # 'result' contains:
    #  - "segmentation": a 2D instance map of shape (H, W) where each pixel's value is the instance id.
    #  - "segments_info": a list of dicts with keys like "id", "label_id", "score".
    instance_map = result["segmentation"]
    segments_info = result["segments_info"]
    
    if isinstance(instance_map, torch.Tensor):
        instance_map = instance_map.cpu().numpy()
    
    print("Instance map shape:", instance_map.shape)
    for seg in segments_info:
        print(f"  Instance {seg['id']} - label_id: {seg.get('label_id', 'N/A')}, score: {seg['score']:.3f}")
    
    # Visualize predictions for this image.
    title = f"Image {img_id} - {file_name}"
    # visualize_prediction(image, instance_map, segments_info, coco_cat_map, model_to_coco, title=title)
    
    # Convert predictions into COCO format.
    for seg in segments_info:
        inst_id = seg["id"]
        model_label = seg.get("label_id", None)
        if model_label is None:
            continue
        coco_label = model_to_coco.get(int(model_label), None)
        if coco_label is None:
            continue
        binary_mask = (instance_map == inst_id).astype(np.uint8)
        if np.sum(binary_mask) == 0:
            continue
        binary_mask = np.asfortranarray(binary_mask)
        rle = maskUtils.encode(binary_mask)
        if rle is None or rle.get("counts") is None:
            continue
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("utf-8")
        prediction = {
            "image_id": img_id,
            "category_id": coco_label,
            "segmentation": rle,
            "score": float(seg["score"])
        }
        predictions.append(prediction)

print(f"\nTotal predictions generated: {len(predictions)}")

# ---------------------------
# (Optional) Evaluate predictions using COCOeval.
# Uncomment the following lines if you want to run evaluation.
from pycocotools.cocoeval import COCOeval
coco_dt = coco_gt.loadRes(predictions)
coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
