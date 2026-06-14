import argparse
import json
import os
import yaml

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from torchvision.ops import nms

from model import YOLOv3

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    

def load_class_names(config):
    """
    Ưu tiên đọc class names từ config.
    Nếu config chưa có, đọc từ train_json hoặc val_json.
    """
    if "classes" in config["data"]:
        return config["data"]["classes"]

    for json_key in ["val_json", "train_json"]:
        json_path = config["data"].get(json_key, None)

        if json_path is not None and os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "classes" in data:
                return data["classes"]

    num_classes = config["data"]["num_classes"]

    return [f"class_{i}" for i in range(num_classes)]

def sigmoid(x):
    return torch.sigmoid(x)

def letterbox_image(image, image_size):
    """
    Resize giữ tỉ lệ + padding về image_size × image_size.
    Phải giống preprocessing lúc train.

    Return:
        canvas: ảnh đã letterbox
        meta: thông tin để map bbox về ảnh gốc
    """
    original_w, original_h = image.size

    scale = min(
        image_size / original_w,
        image_size / original_h
    )

    new_w = int(round(original_w * scale))
    new_h = int(round(original_h * scale))

    resized_image = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new(
        "RGB",
        (image_size, image_size),
        color=(128, 128, 128)
    )

    pad_x = (image_size - new_w) // 2
    pad_y = (image_size - new_h) // 2

    canvas.paste(resized_image, (pad_x, pad_y))

    meta = {
        "original_w": original_w,
        "original_h": original_h,
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "image_size": image_size,
    }

    return canvas, meta

def image_to_tensor(image):
    """
    PIL Image RGB -> torch tensor [1, 3, H, W]
    """
    image_np = np.array(image, dtype=np.float32) / 255.0

    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def prepare_anchors(anchors, image_size, device):
    """
    anchors ban đầu trong config đang ở pixel theo ảnh 416×416.

    Return:
        anchors_normalized shape [3, 3, 2]
    """
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)

    if anchors.max() > 1.0:
        anchors = anchors / image_size

    return anchors


def decode_single_scale(
    prediction,
    anchors_scale,
    conf_thresh,
):
    """
    Decode prediction của một scale.

    prediction:
        [1, 3, S, S, 5 + C]

    anchors_scale:
        [3, 2], normalized theo ảnh, ví dụ [116/416, 90/416]

    Return:
        list detections, mỗi detection là dict:
        {
            "class_id": int,
            "score": float,
            "box": [x1, y1, x2, y2] normalized trên ảnh letterbox
        }
    """
    prediction = prediction[0]

    device = prediction.device

    num_anchors, S, _, pred_dim = prediction.shape

    num_classes = pred_dim - 5

    objectness = sigmoid(prediction[..., 0])

    xy = sigmoid(prediction[..., 1:3])

    wh_raw = prediction[..., 3:5].clamp(min=-10, max=10)

    anchors_scale = anchors_scale.reshape(3, 1, 1, 2)

    wh = torch.exp(wh_raw) * anchors_scale

    class_probs = sigmoid(prediction[..., 5:])

    best_class_probs, best_class_ids = torch.max(
        class_probs,
        dim = -1
    )

    scores = objectness * best_class_probs

    grid_y, grid_x = torch.meshgrid(
        torch.arange(S, device=device),
        torch.arange(S, device=device),
        indexing="ij"
    )

    grid = torch.stack([grid_x, grid_y], dim=-1).float()

    xy = (xy + grid.unsqueeze(0)) / S

    x_center = xy[..., 0]
    y_center = xy[..., 1]
    width = wh[..., 0]
    height = wh[..., 1]

    x1 = x_center - width / 2
    y1 = y_center - height / 2

    x2 = x_center + width / 2
    y2 = y_center + height / 2

    x1 = x1.clamp(0, 1)
    y1 = y1.clamp(0, 1)
    x2 = x2.clamp(0, 1)
    y2 = y2.clamp(0, 1)

    mask = scores > conf_thresh

    detections = []

    selected_scores = scores[mask]
    selected_classes = best_class_ids[mask]
    selected_x1 = x1[mask]
    selected_y1 = y1[mask]
    selected_x2 = x2[mask]
    selected_y2 = y2[mask]

    for idx in range(selected_scores.numel()):
        detections.append({
            "class_id": int(selected_classes[idx].item()),
            "score": float(selected_scores[idx].item()),
            "box": [
                float(selected_x1[idx].item()),
                float(selected_y1[idx].item()),
                float(selected_x2[idx].item()),
                float(selected_y2[idx].item()),
            ]
        })

    return detections

def decode_predictions(
    outputs,
    anchors_normalized,
    conf_thresh,
):
    """
    Decode cả 3 scales.

    outputs:
        [
            [1, 3, 13, 13, 10],
            [1, 3, 26, 26, 10],
            [1, 3, 52, 52, 10],
        ]

    anchors_normalized:
        [3, 3, 2]
    """
    all_detections = []

    for scale_idx in range(3):
        detections = decode_single_scale(
            prediction=outputs[scale_idx],
            anchors_scale=anchors_normalized[scale_idx],
            conf_thresh=conf_thresh,
        )

        all_detections.extend(detections)

    return all_detections

def box_iou(box1, box2, eps=1e-6):
    """
    box1, box2:
        [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)

    intersection = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - intersection + eps

    return intersection / union

def nms_classwise(detections, iou_thresh):
    """
    NMS tự cài đặt, chạy riêng theo từng class.
    """
    final_detections = []

    class_ids = sorted(set(det["class_id"] for det in detections))

    for class_id in class_ids:
        class_dets = [
            det for det in detections
            if det["class_id"] == class_id
        ]

        class_dets = sorted(
            class_dets,
            key=lambda x: x["score"],
            reverse=True
        )

        # Chọn ra những det tốt nhất
        while len(class_dets) > 0:
            best_det = class_dets.pop(0)

            final_detections.append(best_det)

            remaining = []

            for det in class_dets:
                # Tính iou giữa best det và các det còn lại
                iou = box_iou(best_det["box"], det["box"])
                
                if iou < iou_thresh:
                    remaining.append(det)

            class_dets = remaining

    final_detections = sorted(
        final_detections,
        key=lambda x: x["score"],
        reverse=True
    )

    return final_detections

def nms_classwise_torch(detections, iou_thresh, device=None):
    """
    Fast class-wise NMS using torchvision.ops.nms.
    detections: list of dicts with:
        class_id, score, box
    box is [x1, y1, x2, y2], normalized or pixel both OK as long as consistent.
    """
    if len(detections) == 0:
        return []

    if device is None:
        device = torch.device("cpu")

    final_detections = []

    class_ids = sorted(set(det["class_id"] for det in detections))

    for class_id in class_ids:
        class_dets = [
            det for det in detections
            if det["class_id"] == class_id
        ]

        boxes = torch.tensor(
            [det["box"] for det in class_dets],
            dtype=torch.float32,
            device=device,
        )

        scores = torch.tensor(
            [det["score"] for det in class_dets],
            dtype=torch.float32,
            device=device,
        )

        keep_indices = nms(
            boxes=boxes,
            scores=scores,
            iou_threshold=iou_thresh,
        )

        for idx in keep_indices.cpu().tolist():
            final_detections.append(class_dets[idx])

    final_detections = sorted(
        final_detections,
        key=lambda x: x["score"],
        reverse=True,
    )

    return final_detections


def map_box_to_original_image(box, meta):
    """
    box đang normalized trên ảnh letterbox image_size×image_size.

    Chuyển về tọa độ pixel trên ảnh gốc.
    """
    image_size = meta["image_size"]
    pad_x = meta["pad_x"]
    pad_y = meta["pad_y"]
    scale = meta["scale"]
    original_w = meta["original_w"]
    original_h = meta["original_h"]

    x1, y1, x2, y2 = box

    x1 = x1 * image_size
    y1 = y1 * image_size
    x2 = x2 * image_size
    y2 = y2 * image_size

    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    x1 = max(0.0, min(x1, original_w))
    y1 = max(0.0, min(y1, original_h))
    x2 = max(0.0, min(x2, original_w))
    y2 = max(0.0, min(y2, original_h))

    return [x1, y1, x2, y2]

def draw_detections(image, detections, class_names):
    """
    Vẽ bbox lên ảnh gốc.
    """
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box_original"]
        class_id = det["class_id"]
        score = det["score"]

        class_name = class_names[class_id]

        label = f"{class_name} {score:.2f}"

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )

        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        draw.rectangle(
            [x1, y1 - text_h - 4, x1 + text_w + 4, y1],
            fill="red"
        )

        draw.text(
            (x1 + 2, y1 - text_h - 2),
            label,
            fill="white",
            font=font
        )

    return image


def load_model(checkpoint_path, config, device):
    num_classes = config["data"]["num_classes"]

    model = YOLOv3(
        in_channels=3,
        num_classes=num_classes
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model

@torch.no_grad()
def run_inference(
    image_path,
    output_path,
    config,
    checkpoint_path,
    conf_thresh,
    iou_thresh,
):
    device_name = config["train"]["device"]

    if device_name == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA không khả dụng, dùng CPU.")
    else:
        device = torch.device(device_name)

    
    image_size = config["data"]["image_size"]
    anchors = config["anchors"]
    class_names = load_class_names(config)

    model = load_model(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device
    )

    anchors_normalized = prepare_anchors(
        anchors=anchors,
        image_size=image_size,
        device=device
    )

    original_image = Image.open(image_path).convert("RGB")

    letterboxed_image, meta = letterbox_image(
        original_image,
        image_size=image_size
    )

    image_tensor = image_to_tensor(letterboxed_image).to(device)

    outputs = model(image_tensor)

    print("\n===== RAW OUTPUT DEBUG =====")
    for scale_idx, output in enumerate(outputs):
        # output shape: [1, 3, S, S, 10]
        obj = torch.sigmoid(output[..., 0])
        cls = torch.sigmoid(output[..., 5:])

        best_cls_prob, best_cls_id = cls.max(dim=-1)
        score = obj * best_cls_prob

        print(f"Scale {scale_idx}")
        print("  max objectness:", obj.max().item())
        print("  mean objectness:", obj.mean().item())
        print("  max class prob:", best_cls_prob.max().item())
        print("  max final score:", score.max().item())

    detections = decode_predictions(
        outputs=outputs,
        anchors_normalized=anchors_normalized,
        conf_thresh=conf_thresh
    )

    # detections = nms_classwise(
    #     detections=detections,
    #     iou_thresh=iou_thresh
    # )

    detections = nms_classwise_torch(
        detections=detections,
        iou_thresh=iou_thresh,
        device=device,
    )

    for det in detections:
        det["box_original"] = map_box_to_original_image(
            det["box"],
            meta
        )
    
    output_image = original_image.copy()

    output_image = draw_detections(
        output_image,
        detections,
        class_names
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_image.save(output_path)

    print(f"Image path: {image_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Number of detections: {len(detections)}")
    print(f"Saved output image to: {output_path}")

    for det in detections:
        class_name = class_names[det["class_id"]]
        score = det["score"]
        box = det["box_original"]

        print(
            f"{class_name}: {score:.4f} | "
            f"box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/inference_result.jpg"
    )

    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5
    )

    return parser.parse_args()

def main():
    args = parse_args()

    config = load_config(args.config)

    if args.checkpoint is None:
        checkpoint_path = os.path.join(
            config["project"]["save_dir"],
            config["project"]["best_model_name"]
        )
    else:
        checkpoint_path = args.checkpoint

    run_inference(
        image_path=args.image,
        output_path=args.output,
        config=config,
        checkpoint_path=checkpoint_path,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
    )

if __name__ == "__main__":
    main()

