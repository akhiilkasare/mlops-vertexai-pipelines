import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval

# ---------------------------
# Configuration - update these paths accordingly.
# ---------------------------
annotations_file = "/Users/akhilarvindkasare/Downloads/segmentation-model/scripts/coco_sample/annotations/annotations/instances_val2017.json"  # Path to COCO validation annotations
images_dir = "/Users/akhilarvindkasare/Downloads/segmentation-model/scripts/coco_sample/images"  # Directory with images
subset_size = 100  # Number of images to process for testing
mask_threshold = 0.3  # Threshold for binarizing masks

# Output folder to save segmented images
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# ---------------------------
# Load COCO Ground Truth and Build Category Mapping
# ---------------------------
coco_gt = COCO(annotations_file)
if "categories" not in coco_gt.dataset:
    raise ValueError("The annotation file does not contain 'categories' information.")

# Build a dictionary mapping COCO category id to category name.
coco_cat_map = {cat["id"]: cat["name"] for cat in coco_gt.dataset["categories"]}
print("COCO Ground Truth Categories:")
print(coco_cat_map)

# ---------------------------
# Create mapping from model's predicted label IDs to COCO category IDs.
# ---------------------------
def create_model_to_coco_mapping(model, coco_gt):
    model_id2label = model.config.id2label  # e.g., {0: "person", 1: "bicycle", ...}
    gt_categories = coco_gt.dataset["categories"]
    gt_name_to_id = {cat["name"].lower(): cat["id"] for cat in gt_categories}

    custom_mapping = {
         "motorbike": "motorcycle",
         "aeroplane": "airplane",
         "sofa": "couch",
         "pottedplant": "potted plant",
         "diningtable": "dining table",
         "tvmonitor": "tv"
    }

    mapping = {}
    for model_label, model_cat_name in model_id2label.items():
         model_cat_name_lower = model_cat_name.lower()
         if model_cat_name_lower in custom_mapping:
              model_cat_name_lower = custom_mapping[model_cat_name_lower]
         gt_cat_id = gt_name_to_id.get(model_cat_name_lower, None)
         if gt_cat_id is not None:
              mapping[int(model_label)] = int(gt_cat_id)
         else:
              print(f"Warning: Model label {model_label} with name '{model_cat_name}' not found in COCO ground truth.")
    return mapping

# ---------------------------
# Diagnostic Functions (omitted here for brevity; see previous code for full versions)
# ---------------------------
def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0

def validate_iou_overlap(coco_gt, predictions, image_id, iou_threshold=0.5):
    ann_ids = coco_gt.getAnnIds(imgIds=image_id)
    gt_anns = coco_gt.loadAnns(ann_ids)
    gt_masks = [(ann['category_id'], coco_gt.annToMask(ann)) for ann in gt_anns]
    
    preds = [p for p in predictions if p['image_id'] == image_id]
    pred_masks = []
    for p in preds:
        rle = p['segmentation']
        pred_mask = maskUtils.decode(rle)
        pred_masks.append((p['category_id'], pred_mask, p['score']))
    
    if pred_masks:
        all_masks = [pm for (_, pm, _) in pred_masks]
        combined = np.stack(all_masks, axis=0)
        print("Unique pixel values across predicted masks (should be binary 0/1):", np.unique(combined))
    
    iou_list = []
    for cat_id, pred_mask, score in pred_masks:
        max_iou = 0.0
        for gt_cat_id, gt_mask in gt_masks:
            if gt_cat_id == cat_id:
                iou = compute_iou(pred_mask, gt_mask)
                max_iou = max(max_iou, iou)
        iou_list.append(max_iou)
    
    print(f"Image {image_id} - Number of predictions: {len(pred_masks)}")
    if iou_list:
        print(f"Mean IoU for predictions: {np.mean(iou_list):.3f}")
        print(f"Max IoU for predictions: {np.max(iou_list):.3f}")
        print(f"Min IoU for predictions: {np.min(iou_list):.3f}")
        num_valid = sum(iou >= iou_threshold for iou in iou_list)
        print(f"Number of predictions with IoU >= {iou_threshold}: {num_valid}")
    else:
        print("No predictions for this image.")
    scores = [score for _,_,score in pred_masks]
    if scores:
        print(f"Mean confidence: {np.mean(scores):.3f}, Max: {np.max(scores):.3f}, Min: {np.min(scores):.3f}")
    else:
        print("No prediction scores available.")

def validate_category_mapping(model, coco_gt, custom_mapping=None):
    model_id2label = model.config.id2label
    gt_categories = coco_gt.dataset["categories"]
    gt_names = set([cat["name"].lower() for cat in gt_categories])
    
    if custom_mapping is None:
        custom_mapping = {
            "motorbike": "motorcycle",
            "aeroplane": "airplane",
            "sofa": "couch",
            "pottedplant": "potted plant",
            "diningtable": "dining table",
            "tvmonitor": "tv"
        }
    
    for label_id, label_name in model_id2label.items():
        label_name_lower = label_name.lower()
        mapped_name = custom_mapping.get(label_name_lower, label_name_lower)
        if mapped_name not in gt_names:
            print(f"Warning: Model label {label_id} with name '{label_name}' (mapped to '{mapped_name}') not in COCO ground truth.")

def evaluate_threshold_effects(processor, model, image, target_size, thresholds=[0.05, 0.1, 0.3, 0.5]):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = {}
    for t in thresholds:
        result = processor.post_process_instance_segmentation(outputs, threshold=t, target_sizes=[target_size])[0]
        num_segments = len(result["segments_info"])
        results[t] = result
        print(f"Threshold {t}: {num_segments} segments detected.")
    return results

def plot_score_distribution(predictions):
    scores = [pred["score"] for pred in predictions]
    if not scores:
        print("No scores to plot.")
        return
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=20, edgecolor='black')
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Confidence Scores")
    plt.show()

# ---------------------------
# Functions to Save Visualizations
# ---------------------------
def save_segmentation_visualization(img, instance_map, segments_info, coco_cat_map, model_to_coco, title, save_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img)
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
            ax.text(int(np.mean(xs)), int(np.mean(ys)), f"{label_name}: {score:.2f}", fontsize=10,
                    color="white", bbox=dict(facecolor="red", alpha=0.5))
    ax.imshow(overlay, alpha=0.5)
    ax.set_title(title)
    ax.axis("off")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Segmentation visualization saved to: {save_path}")

def save_ground_truth_visualization(img, coco_gt, image_id, save_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img)
    ann_ids = coco_gt.getAnnIds(imgIds=image_id)
    gt_anns = coco_gt.loadAnns(ann_ids)
    for ann in gt_anns:
        gt_mask = coco_gt.annToMask(ann)
        contours = np.where(gt_mask == 1)
        if contours[0].size > 0:
            y, x = np.mean(contours[0]), np.mean(contours[1])
            ax.text(x, y, f"GT: {ann['category_id']}", color="yellow", fontsize=12,
                    bbox=dict(facecolor="black", alpha=0.5))
    ax.set_title("Ground Truth Annotations")
    ax.axis("off")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Ground truth visualization saved to: {save_path}")

# ---------------------------
# Load the pretrained Mask2Former model and its processor.
# For consistency, we force use_fast=True.
# ---------------------------
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-instance", use_fast=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-instance")
model.eval()

# Validate category mapping.
validate_category_mapping(model, coco_gt)

# Create mapping from model label to COCO category.
model_to_coco = create_model_to_coco_mapping(model, coco_gt)
print("*" * 50)
print("Model to COCO Mapping:")
print(model_to_coco)
print("*" * 50)

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
    
    # Ensure the target size is the original image size (height, width)
    target_size = image.size[::-1]
    print("Target size:", target_size)
    
    # Postprocess outputs using the specified mask threshold.
    result = processor.post_process_instance_segmentation(outputs, threshold=mask_threshold, target_sizes=[target_size])[0]
    
    instance_map = result["segmentation"]
    segments_info = result["segments_info"]
    
    if isinstance(instance_map, torch.Tensor):
        instance_map = instance_map.cpu().numpy()
    
    print("Instance map shape:", instance_map.shape)
    print("Unique instance IDs in segmentation:", np.unique(instance_map))
    
    for seg in segments_info:
        binary_mask = (instance_map == seg["id"]).astype(np.uint8)
        area = np.sum(binary_mask)
        print(f"  Instance {seg['id']} - label_id: {seg.get('label_id', 'N/A')}, score: {seg['score']:.3f}, mask area: {area}")
    
    # Save the segmentation visualization for this image.
    seg_save_path = os.path.join(output_folder, f"segmentation_{img_id}.png")
    save_segmentation_visualization(image, instance_map, segments_info, coco_cat_map, model_to_coco,
                                    f"Segmentation for Image {img_id}", seg_save_path)
    
    # (Optional) Save ground truth visualization as well.
    gt_save_path = os.path.join(output_folder, f"ground_truth_{img_id}.png")
    save_ground_truth_visualization(image, coco_gt, img_id, gt_save_path)
    
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
# Evaluate predictions using COCOeval.
# ---------------------------
coco_dt = coco_gt.loadRes(predictions)
coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# ---------------------------
# Additional Diagnostics:
# ---------------------------
# 1. Evaluate threshold effects on a sample image.
sample_image_id = selected_ids[0]
sample_img_info = coco_gt.loadImgs(sample_image_id)[0]
sample_img_path = os.path.join(images_dir, sample_img_info["file_name"])
sample_image = Image.open(sample_img_path).convert("RGB")
sample_target_size = sample_image.size[::-1]
print("\nEvaluating threshold effects on a sample image:")
evaluate_threshold_effects(processor, model, sample_image, sample_target_size, thresholds=[0.05, 0.1, 0.3, 0.5])

# 2. Validate IoU overlap for the sample image.
print("\nValidating IoU overlap for the sample image:")
validate_iou_overlap(coco_gt, predictions, sample_image_id, iou_threshold=0.05)

# 3. Plot the confidence score distribution.
print("\nPlotting confidence score distribution:")
plot_score_distribution(predictions)
