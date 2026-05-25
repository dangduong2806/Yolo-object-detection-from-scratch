import os
import random
import yaml
import torch
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from model import YOLOv3
from loss import YOLOv3Loss
from dataset import YOLODataset

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

        running_loss += loss.items()

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
        config,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config
    }

    torch.save(checkpoint, save_path)


def main():
    config = load_config("config.yaml")

    seed = config["train"]["seed"]
    set_seed(seed=seed)

    requested_device = config["train"]["device"]

    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA không khả dụng, chuyển sang CPU.")
    else:
        device = torch.device(requested_device)
    
    save_dir = config["project"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    best_model_path = os.path.join(
        save_dir,
        config["project"]["best_model_name"]
    )

    last_model_path = os.path.join(
        save_dir,
        config["project"]["last_model_name"]
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

    train_dataset = YOLODataset(
        json_path=config["data"]["train_json"],
        root_dir=config["data"]["root_dir"],
        anchors=anchors,
        image_size=image_size,
        scales=scales,
        ignore_iou_thresh=config["data"]["ignore_iou_thresh"],
        use_letterbox=config["data"]["use_letterbox"]
    )

    val_dataset = YOLODataset(
        json_path=config["data"]["val_json"],
        root_dir=config["data"]["root_dir"],
        anchors=anchors,
        image_size=image_size,
        scales=scales,
        ignore_iou_thresh=config["data"]["ignore_iou_thresh"],
        use_letterbox=config["data"]["use_letterbox"],
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

    model = YOLOv3(
        in_channels=3,
        num_classes=num_classes
    ).to(device=device)

    loss_fn = YOLOv3Loss(
        num_classes=num_classes,
        lambda_box=config["loss"]["lambda_box"],
        lambda_obj=config["loss"]["lambda_obj"],
        lambda_noobj=config["loss"]["lambda_noobj"],
        lambda_class=config["loss"]["lambda_class"],
    )

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )

    use_amp = (
        config["train"]["mixed_precision"]
        and device.type == "cuda"
    )

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_val_loss = float("inf")

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

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}"
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
        
        save_checkpoint(
            save_path=last_model_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            config=config,
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                save_path=best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                config=config,
            )

            print(
                f"Saved best model: {best_model_path} "
                f"with val_loss={best_val_loss:.4f}"
            )

    print("\nTraining completed.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()