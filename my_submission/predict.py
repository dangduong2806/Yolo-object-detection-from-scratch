import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "utils"))

from inference import (  # noqa: E402
    apply_class_conf_thresholds,
    decode_predictions,
    image_to_tensor,
    letterbox_image,
    limit_detections,
    load_class_names,
    load_config,
    load_model,
    map_box_to_original_image,
    nms_classwise_torch,
    prepare_anchors,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="models/best.pth")
    parser.add_argument("--weights-url", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--conf-thresh", type=float, default=0.30)
    parser.add_argument("--nms-iou-thresh", type=float, default=0.30)
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def list_images(image_dir):
    image_dir = Path(image_dir)
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_weights_url(args, config):
    return (
        args.weights_url
        or os.environ.get("MODEL_WEIGHTS_URL")
        or config.get("weights", {}).get("url")
    )


def ensure_checkpoint_exists(checkpoint_path, weights_url):
    if checkpoint_path.exists():
        return

    if not weights_url:
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Provide --weights-url, set MODEL_WEIGHTS_URL, or set weights.url in config.yaml."
        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".download")

    print(f"Checkpoint not found. Downloading weights from: {weights_url}")
    try:
        urllib.request.urlretrieve(weights_url, temp_path)
        temp_path.replace(checkpoint_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


@torch.no_grad()
def main():
    args = parse_args()
    config = load_config(resolve_path(args.config))

    if args.device is not None:
        config["train"]["device"] = args.device

    if args.max_detections is not None:
        config.setdefault("postprocess", {})["max_detections_per_image"] = args.max_detections

    requested_device = config["train"].get("device", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    checkpoint_path = resolve_path(args.checkpoint)
    ensure_checkpoint_exists(
        checkpoint_path=checkpoint_path,
        weights_url=get_weights_url(args, config),
    )

    model = load_model(
        checkpoint_path=str(checkpoint_path),
        config=config,
        device=device,
    )

    class_names = load_class_names(config)
    anchors_normalized = prepare_anchors(
        anchors=config["anchors"],
        image_size=config["data"]["image_size"],
        device=device,
    )

    postprocess_config = config.get("postprocess", {})
    class_thresholds = postprocess_config.get("class_conf_thresholds", {})
    max_detections = postprocess_config.get("max_detections_per_image", args.max_detections)

    predictions = []
    image_paths = list_images(args.image_dir)

    for image_path in tqdm(image_paths, desc="Predict"):
        original_image = Image.open(image_path).convert("RGB")
        letterboxed_image, meta = letterbox_image(
            original_image,
            image_size=config["data"]["image_size"],
        )

        image_tensor = image_to_tensor(
            letterboxed_image,
            imagenet_normalize=config.get("model", {}).get("imagenet_normalize", False),
        ).to(device)

        outputs = model(image_tensor)
        detections = decode_predictions(
            outputs=outputs,
            anchors_normalized=anchors_normalized,
            conf_thresh=args.conf_thresh,
        )

        if postprocess_config.get("enabled", False):
            detections = apply_class_conf_thresholds(
                detections=detections,
                class_names=class_names,
                class_conf_thresholds=class_thresholds,
            )

        detections = nms_classwise_torch(
            detections=detections,
            iou_thresh=args.nms_iou_thresh,
            device=device,
        )

        if postprocess_config.get("enabled", False):
            detections = limit_detections(
                detections=detections,
                max_detections=max_detections,
            )

        boxes = []
        for det in detections:
            box_original = map_box_to_original_image(det["box"], meta)
            x1, y1, x2, y2 = box_original
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append({
                "class": class_names[det["class_id"]],
                "confidence": float(max(0.0, min(1.0, det["score"]))),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })

        predictions.append({
            "image_id": image_path.name,
            "boxes": boxes,
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path(".") else None
    output_path.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
