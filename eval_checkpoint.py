# File này để in ra các chỉ số ap@0.5, precision, recall trên từng class
# Mục đích là để đánh giá xem checkpoint tốt nhất hiện tại đang yếu ở đâu
# Từ đó cải tiến nó lên

# AP@0.5 per class: điểm của từng class.
# mAP@0.5 overall: trung bình AP của các class.
    
import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from inference import (
    load_config,
    load_class_names,
    load_model,
    letterbox_image,
    image_to_tensor,
    prepare_anchors,
    decode_predictions,
    nms_classwise,
    map_box_to_original_image,
    nms_classwise_torch
)

from tools.evaluate_predictions import (
    load_json,
    validate_ground_truth,
    normalize_predictions,
    evaluate,
)

from train import build_val_predictions

def compute_metrics(val_json_path, predictions_json, iou_threshold):
    ground_truth = load_json(Path(val_json_path))
    classes, image_info = validate_ground_truth(ground_truth)

    normalized_predictions = normalize_predictions(
        predictions_json,
        classes=classes,
        image_info=image_info,
        max_detections_per_image=100,
        require_complete=True,
    )

    return evaluate(
        ground_truth=ground_truth,
        predictions=normalized_predictions,
        classes=classes,
        iou_threshold=iou_threshold
    )

def print_metrics(metrics):
    print("\n===== Overall =====")
    print(f"mAP@0.5        : {metrics['mAP@0.5']:.6f}")
    print(f"micro precision: {metrics['micro_precision']:.6f}")
    print(f"micro recall   : {metrics['micro_recall']:.6f}")
    print(f"GT boxes       : {metrics['num_ground_truth_boxes']}")
    print(f"Pred boxes     : {metrics['num_predictions']}")

    print("\n===== Per class =====")
    print(f"{'class':<20} {'AP@0.5':>10} {'precision':>10} {'recall':>10} {'GT':>6} {'Pred':>6} {'TP':>6} {'FP':>6}")

    per_class = metrics["per_class"]

    for class_name, item in sorted(per_class.items(), key=lambda x: x[1]["ap"]):
        print(
            f"{class_name:<20} "
            f"{item['ap']:>10.6f} "
            f"{item['precision']:>10.6f} "
            f"{item['recall']:>10.6f} "
            f"{item['num_ground_truth']:>6} "
            f"{item['num_predictions']:>6} "
            f"{item['true_positives']:>6} "
            f"{item['false_positives']:>6}"
        )

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--conf-thresh", type=float, default=0.25)
    parser.add_argument("--nms-iou-thresh", type=float, default=0.5)
    parser.add_argument("--eval-iou-thresh", type=float, default=0.5)
    parser.add_argument("--save-predictions", type=str, default=None)
    parser.add_argument("--save-metrics", type=str, default=None)

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
    )

    return parser.parse_args()

def main():
    args = parse_args()

    config = load_config(args.config)

    if args.split == "train":
        config["data"]["val_json"] = config["data"]["train_json"]

    requested_device = config["train"]["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")
    else:
        device = torch.device(requested_device)

    model = load_model(
        checkpoint_path=args.checkpoint,
        config=config,
        device=device,
    )

    predictions_json = build_val_predictions(
        model=model,
        config=config,
        device=device,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.nms_iou_thresh,
    )

    metrics = compute_metrics(
        val_json_path=config["data"]["val_json"],
        # val_json_path=config["data"]["train_json"],
        predictions_json=predictions_json,
        iou_threshold=args.eval_iou_thresh,
    )

    print_metrics(metrics=metrics)

    if args.save_predictions:
        Path(args.save_predictions).write_text(
            json.dumps(predictions_json, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.save_metrics:
        Path(args.save_metrics).write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    
if __name__ == "__main__":
    main()