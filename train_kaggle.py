import os
import random
import yaml
import torch
import numpy as np
import json

from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from pathlib import Path

from model import YOLOv3
from model_resnet_yolov3 import ResNet50YOLOv3
from loss import YOLOv3Loss
from dataset_kaggle import YOLODataset

from augmentations import DetectionAugmenter

from inference_kaggle import (
    letterbox_image,
    image_to_tensor,
    prepare_anchors,
    decode_predictions,
    nms_classwise,
    map_box_to_original_image,
    load_class_names,
    nms_classwise_torch
)

from tools.evaluate_predictions import (
    load_json,
    validate_ground_truth,
    normalize_predictions,
    evaluate,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def prepare_scaled_anchors(anchors, image_size, scales, device):
    """
    anchors ban đầu đang ở pixel theo ảnh 416x416.

    Ta cần đưa anchors về đơn vị cell của từng scale.

    Ví dụ:
        anchor [116, 90] / 416 = [0.279, 0.216]
        scale 13:
        [0.279, 0.216] * 13 = [3.63, 2.81]
    """
    anchors = torch.tensor(anchors, dtype=torch.float32)

    anchors_normalized = anchors / image_size

    scales_tensor = torch.tensor(
        scales,
        dtype=torch.float32
    ).reshape(3, 1, 1)

    scaled_anchors = anchors_normalized * scales_tensor

    return scaled_anchors.to(device)

def move_targets_to_device(targets, device):
    """
    DataLoader sẽ collate targets thành list/tuple gồm 3 tensor:
        targets[0]: [B, 3, 13, 13, 6]
        targets[1]: [B, 3, 26, 26, 6]
        targets[2]: [B, 3, 52, 52, 6]
    """
    return [target.to(device) for target in targets]

def compute_yolo_loss(outputs, targets, scaled_anchors, loss_fn):
    """
    outputs:
        outputs[0]: [B, 3, 13, 13, 10]
        outputs[1]: [B, 3, 26, 26, 10]
        outputs[2]: [B, 3, 52, 52, 10]

    targets:
        targets[0]: [B, 3, 13, 13, 6]
        targets[1]: [B, 3, 26, 26, 6]
        targets[2]: [B, 3, 52, 52, 6]
    """
    total_loss = 0.0

    loss_log = {
        "box_loss": 0.0,
        "obj_loss": 0.0,
        "noobj_loss": 0.0,
        "class_loss": 0.0
    }

    for scale_idx in range(3):
        scale_loss, loss_items = loss_fn(
            outputs[scale_idx],
            targets[scale_idx],
            scaled_anchors[scale_idx]
        )

        total_loss = total_loss + scale_loss

        loss_log["box_loss"] += loss_items["box_loss"].item()
        loss_log["obj_loss"] += loss_items["obj_loss"].item()
        loss_log["noobj_loss"] += loss_items["noobj_loss"].item()
        loss_log["class_loss"] += loss_items["class_loss"].item()

    return total_loss, loss_log

def train_one_epoch(
        model,
        train_loader,
        optimizer,
        loss_fn,
        scaled_anchors,
        device,
        scaler=None,
        gradient_clip_norm=None,
):
    model.train()

    running_loss = 0.0
    running_logs = {
        "box_loss": 0.0,
        "obj_loss": 0.0,
        "noobj_loss": 0.0,
        "class_loss": 0.0
    }

    progess_bar = tqdm(train_loader, desc="Training", leave=False)

    for batch_idx, (images, targets) in enumerate(progess_bar):
        images = images.to(device)
        targets = move_targets_to_device(targets=targets, device=device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss, loss_log = compute_yolo_loss(
                    outputs=outputs,
                    targets=targets,
                    scaled_anchors=scaled_anchors,
                    loss_fn=loss_fn
                )
            
            scaler.scale(loss).backward()

            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip_norm
                )
            
            scaler.step(optimizer)
            scaler.update()
        
        else:
            outputs = model(images)
            loss, loss_log = compute_yolo_loss(
                outputs=outputs,
                targets=targets,
                scaled_anchors=scaled_anchors,
                loss_fn=loss_fn
            )

            loss.backward()

            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip_norm
                )
            
            optimizer.step()
        
        running_loss += loss.item()

        for key in running_logs:
            running_logs[key] += loss_log[key]
        
        avg_loss = running_loss / (batch_idx + 1)

        progess_bar.set_postfix({
            "loss": f"{avg_loss:.4f}"
        })
    
    num_batches = len(train_loader)

    epoch_loss = running_loss / num_batches

    epoch_logs = {
        key: value / num_batches
        for key, value in running_logs.items()
    }

    return epoch_loss, epoch_logs

@torch.no_grad()
def build_val_predictions(
    model,
    config,
    device,
    conf_thresh=0.25,
    iou_thresh=0.5,
):
    model.eval()
    root_dir = config["data"]["root_dir"]
    val_json = config["data"]["val_json"]
    # val_json = config["data"]["train_json"]
    image_size = config["data"]["image_size"]
    anchors = config["anchors"]

    class_names = load_class_names(config)

    anchors_normalized = prepare_anchors(
        anchors=anchors,
        image_size=image_size,
        device=device,
    )

    with open(val_json, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    
    predictions_json = []

    for image_info in val_data["images"]:
        image_id = image_info["id"]
        file_name = image_info["file_name"]

        image_path = os.path.join(root_dir, file_name)

        original_image = Image.open(image_path).convert("RGB")

        letterboxed_image, meta = letterbox_image(
            original_image,
            image_size=image_size,
        )

        # image_tensor = image_to_tensor(letterboxed_image).to(device)
        image_tensor = image_to_tensor(
            letterboxed_image,
            imagenet_normalize=config.get("model", {}).get("imagenet_normalize", False),
        ).to(device)

        outputs = model(image_tensor)

        detections = decode_predictions(
            outputs=outputs,
            anchors_normalized=anchors_normalized,
            conf_thresh=conf_thresh,
        )

        # detections = nms_classwise(
        #     detections=detections,
        #     iou_thresh=iou_thresh,
        # )
        detections = nms_classwise_torch(
            detections=detections,
            iou_thresh=iou_thresh,
            device=device,
        )

        boxes = []

        for det in detections:
            box_original = map_box_to_original_image(
                det["box"],
                meta,
            )

            x1, y1, x2, y2 = box_original
            if x2 <= x1 or y2 <= y1:
                continue

            class_id = det["class_id"]

            boxes.append({
                "class": class_names[class_id],
                "confidence": float(det["score"]),
                "bbox": [
                    float(box_original[0]),
                    float(box_original[1]),
                    float(box_original[2]),
                    float(box_original[3]),
                ],
            })

        predictions_json.append({
            "image_id": image_id,
            "boxes": boxes,
        })

    return predictions_json

def compute_map50_from_predictions(
        val_json_path,
        predictions_json,
        iou_threshold=0.5,
):
    ground_truth = load_json(Path(val_json_path))

    classes, image_info = validate_ground_truth(ground_truth)

    normalized_predictions = normalize_predictions(
        predictions_json,
        classes=classes,
        image_info=image_info,
        max_detections_per_image=100,
        require_complete=True,
    )

    metrics = evaluate(
        ground_truth=ground_truth,
        predictions=normalized_predictions,
        classes=classes,
        iou_threshold=iou_threshold,
    )

    return metrics

@torch.no_grad()
def validate_one_epoch(
    model,
    val_loader,
    loss_fn,
    scaled_anchors,
    device,
):
    model.eval()

    running_loss = 0.0

    running_logs = {
        "box_loss": 0.0,
        "obj_loss": 0.0,
        "noobj_loss": 0.0,
        "class_loss": 0.0,
    }

    progress_bar = tqdm(val_loader, desc="Validation", leave=False)
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = move_targets_to_device(targets=targets, device=device)

        outputs = model(images)

        loss, loss_log = compute_yolo_loss(
            outputs=outputs,
            targets=targets,
            scaled_anchors=scaled_anchors,
            loss_fn=loss_fn
        )

        running_loss += loss.item()

        for key in running_logs:
            running_logs[key] += loss_log[key]
        
        avg_loss = running_loss / (batch_idx + 1)

        progress_bar.set_postfix({
            "val_loss": f"{avg_loss:.4f}"
        })

    num_batches = len(val_loader)

    epoch_loss = running_loss / num_batches

    epoch_logs = {
        key: value / num_batches
        for key, value in running_logs.items()
    }

    return epoch_loss, epoch_logs


def save_checkpoint(
        save_path,
        model,
        optimizer,
        epoch,
        train_loss,
        val_loss,
        map50,
        config,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "map50": map50,
        "config": config
    }

    torch.save(checkpoint, save_path)


def main():
    config = load_config("config_kaggle.yaml")

    seed = config["train"]["seed"]
    set_seed(seed=seed)

    requested_device = config["train"]["device"]

    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA không khả dụng, chuyển sang CPU.")
    else:
        device = torch.device(requested_device)

    # Khởi tạo augmentation
    augment_config = config.get("augment", {})
    train_augmenter = (
        DetectionAugmenter(augment_config)
        if augment_config.get("enabled", False)
        else None
    )
    
    save_dir = config["project"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    best_loss_model_path = os.path.join(
        save_dir,
        config["project"]["best_loss_model_name"]
    )

    last_model_path = os.path.join(
        save_dir,
        config["project"]["last_model_name"]
    )

    best_map_model_path = os.path.join(
        save_dir,
        config["project"]["best_map_model_name"]
    )

    image_size = config["data"]["image_size"]
    num_classes = config["data"]["num_classes"]
    scales = config["data"]["scales"]
    anchors = config["anchors"]

    scaled_anchors = prepare_scaled_anchors(
        anchors=anchors,
        image_size=image_size,
        scales=scales,
        device=device
    )

    imagenet_normalize=config.get("model", {}).get("imagenet_normalize", False)

    train_dataset = YOLODataset(
        json_path=config["data"]["train_json"],
        root_dir=config["data"]["root_dir"],
        anchors=anchors,
        image_size=image_size,
        scales=scales,
        ignore_iou_thresh=config["data"]["ignore_iou_thresh"],
        use_letterbox=config["data"]["use_letterbox"],
        augmenter=train_augmenter,
        imagenet_normalize=imagenet_normalize,
    )

    val_dataset = YOLODataset(
        json_path=config["data"]["val_json"],
        root_dir=config["data"]["root_dir"],
        anchors=anchors,
        image_size=image_size,
        scales=scales,
        ignore_iou_thresh=config["data"]["ignore_iou_thresh"],
        use_letterbox=config["data"]["use_letterbox"],
        augmenter=None,
        imagenet_normalize=imagenet_normalize
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    # model = YOLOv3(
    #     in_channels=3,
    #     num_classes=num_classes
    # ).to(device=device)

    model_name = config.get("model", {}).get("name", "yolov3_from_scratch")

    if model_name == "resnet50_yolov3":
        model = ResNet50YOLOv3(
            in_channels=3,
            num_classes=num_classes,
            pretrained=config.get("model", {}).get("pretrained", True),
            freeze_backbone=config.get("model", {}).get("freeze_backbone", False),
        ).to(device=device)
    else:
        model = YOLOv3(
            in_channels=3,
            num_classes=num_classes,
        ).to(device=device)

    loss_fn = YOLOv3Loss(
        num_classes=num_classes,
        lambda_box=config["loss"]["lambda_box"],
        lambda_obj=config["loss"]["lambda_obj"],
        lambda_noobj=config["loss"]["lambda_noobj"],
        lambda_class=config["loss"]["lambda_class"],
        box_loss_type=config["loss"].get("box_loss_type", "mse"),
        obj_label_smoothing=config["loss"].get("obj_label_smoothing", 0.0),
        class_label_smoothing=config["loss"].get("class_label_smoothing", 0.0),
    )

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )

    scheduler_name = config["train"].get("scheduler", None)

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["train"]["epochs"],
            eta_min=config["train"].get("min_learning_rate", 1e-6),
        )
    else:
        scheduler = None

    use_amp = (
        config["train"]["mixed_precision"]
        and device.type == "cuda"
    )

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_val_loss = float("inf")
    best_map50 = 0.0
    best_val_loss_at_best_map = float("inf")

    early_config = config.get("early_stopping", {})
    early_enabled = early_config.get("enabled", False)
    early_monitor = early_config.get("monitor", "map50")
    early_patience = early_config.get("patience", 3)
    early_min_delta = early_config.get("min_delta", 0.001)
    early_start_epoch = early_config.get("start_epoch", 0)
    early_bad_count = 0

    map_eval_interval = config["train"]["map_eval_interval"]

    epochs = config["train"]["epochs"]

    print("Start training YOLOv3 from scratch")
    print(f"Device       : {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples  : {len(val_dataset)}")
    print(f"Save dir     : {save_dir}")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch [{epoch}/{epochs}]")

        train_loss, train_logs = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaled_anchors=scaled_anchors,
            device=device,
            scaler=scaler,
            gradient_clip_norm=config["train"]["gradient_clip_norm"],
        )

        val_loss, val_logs = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            scaled_anchors=scaled_anchors,
            device=device
        )

        should_eval_map = (
            epoch % map_eval_interval == 0
            or epoch == epochs
        )

        map50 = None
        micro_precision = None
        micro_recall = None

        if should_eval_map:
            predictions_json = build_val_predictions(
                model=model,
                config=config,
                device=device,
                conf_thresh=0.25,
                iou_thresh=0.5
            )

            map_metrics = compute_map50_from_predictions(
                val_json_path=config["data"]["val_json"],
                predictions_json=predictions_json,
                iou_threshold=0.5,
            )

            map50 = map_metrics["mAP@0.5"]
            micro_precision = map_metrics["micro_precision"]
            micro_recall = map_metrics["micro_recall"]

        if should_eval_map:
            print(
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"mAP@0.5: {map50:.4f} | "
                f"precision: {micro_precision:.4f} | "
                f"recall: {micro_recall:.4f}"
            )
        else:
            print(
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"mAP@0.5: skipped"
            )

        print(
            "Train details: "
            f"box={train_logs['box_loss']:.4f}, "
            f"obj={train_logs['obj_loss']:.4f}, "
            f"noobj={train_logs['noobj_loss']:.4f}, "
            f"class={train_logs['class_loss']:.4f}"
        )

        print(
            "Val details: "
            f"box={val_logs['box_loss']:.4f}, "
            f"obj={val_logs['obj_loss']:.4f}, "
            f"noobj={val_logs['noobj_loss']:.4f}, "
            f"class={val_logs['class_loss']:.4f}"
        )
        
        # save last model
        save_checkpoint(
            save_path=last_model_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            map50=map50,
            config=config,
        )
        
        # best loss checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                save_path=best_loss_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                map50=map50,
                config=config,
            )

            print(
                f"Saved best loss model: {best_loss_model_path} "
                f"with best val_loss={best_val_loss:.4f}"
            )

        # best MAP checkpoint
        if should_eval_map:
            eps = 1e-4

            previous_best_map50 = best_map50

            is_best_map = (
                map50 > best_map50 + eps
                or (
                    abs(map50 - best_map50) <= eps
                    and val_loss < best_val_loss_at_best_map
                )
            )

            if is_best_map:
                best_map50 = map50
                best_val_loss_at_best_map = val_loss

                save_checkpoint(
                    save_path=best_map_model_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=best_val_loss_at_best_map,
                    map50=best_map50,
                    config=config,
                )
            
            if early_enabled and epoch >= early_start_epoch:
                improved_for_early_stop = map50 > previous_best_map50 + early_min_delta

                if improved_for_early_stop:
                    early_bad_count = 0
                else:
                    early_bad_count += 1

                print(
                    f"Early stopping: bad_count={early_bad_count}/"
                    f"{early_patience}, best_mAP@0.5={best_map50:.4f}"
                )

                if early_bad_count >= early_patience:
                    print(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"No mAP improvement > {early_min_delta} "
                        f"for {early_patience} eval rounds."
                    )
                    break

            
        
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"LR: {current_lr:.8f}")

    print("\nTraining completed.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val loss at best MAP@0.5: {best_val_loss_at_best_map:.4f}")
    print(f"Best MAP@0.5 score: {best_map50:.4f}")
    print(f"Best model saved at: {best_map_model_path}")


if __name__ == "__main__":
    main()