import argparse
import os

import torch
from torch.utils.data import DataLoader

from model import YOLOv3
from model_resnet_yolov3 import ResNet50YOLOv3
from loss import YOLOv3Loss

try:
    from dataset_kaggle import YOLODataset
except ImportError:
    from dataset import YOLODataset

from train_kaggle import (
    load_config,
    set_seed,
    prepare_scaled_anchors,
    train_one_epoch,
    validate_one_epoch,
    build_val_predictions,
    compute_map50_from_predictions,
)

try:
    from augmentations import DetectionAugmenter
except ImportError:
    DetectionAugmenter = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resume Kaggle training from an existing checkpoint."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_kaggle.yaml",
        help="Path to config yaml.",
    )
    return parser.parse_args()


def build_scheduler(config, optimizer):
    scheduler_name = config["train"].get("scheduler", None)

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["train"].get("lr_t_max", config["train"]["epochs"]),
            eta_min=config["train"].get("min_learning_rate", 1e-6),
        )

    return None


def compute_completed_scheduler_steps(config, resumed_epoch):
    """
    Number of scheduler.step() calls that should have happened before
    starting the next epoch.

    Default:
        phase_start_epoch = 1
        resumed_epoch = 20  -> completed steps = 20

    Phase 2 example:
        phase_start_epoch = 6
        resumed_epoch = 20  -> completed steps = 15
        because phase-2 epochs completed are 6..20.
    """
    phase_start_epoch = config["train"].get("phase_start_epoch", 1)
    return max(0, int(resumed_epoch) - int(phase_start_epoch) + 1)


def fast_forward_scheduler(scheduler, num_steps):
    if scheduler is None:
        return

    for _ in range(int(num_steps)):
        scheduler.step()


def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        param_group["initial_lr"] = lr


def build_model(config, device):
    num_classes = config["data"]["num_classes"]
    model_name = config.get("model", {}).get("name", "yolov3_from_scratch")

    if model_name == "resnet50_yolov3":
        # pretrained=False is correct for resume:
        # the checkpoint already contains the trained weights.
        model = ResNet50YOLOv3(
            in_channels=3,
            num_classes=num_classes,
            pretrained=False,
            freeze_backbone=config.get("model", {}).get("freeze_backbone", False),
        ).to(device)
    else:
        model = YOLOv3(
            in_channels=3,
            num_classes=num_classes,
        ).to(device)

    return model


def build_loss(config):
    return YOLOv3Loss(
        num_classes=config["data"]["num_classes"],
        lambda_box=config["loss"]["lambda_box"],
        lambda_obj=config["loss"]["lambda_obj"],
        lambda_noobj=config["loss"]["lambda_noobj"],
        lambda_class=config["loss"]["lambda_class"],
        box_loss_type=config["loss"].get("box_loss_type", "mse"),
        obj_label_smoothing=config["loss"].get("obj_label_smoothing", 0.0),
        class_label_smoothing=config["loss"].get("class_label_smoothing", 0.0),
    )


def save_checkpoint_v2(
    save_path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    train_loss,
    val_loss,
    map50,
    config,
    best_val_loss,
    best_map50,
    best_val_loss_at_best_map,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "map50": map50,
        "best_val_loss": best_val_loss,
        "best_map50": best_map50,
        "best_val_loss_at_best_map": best_val_loss_at_best_map,
        "config": config,
    }

    torch.save(checkpoint, save_path)


def main():
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["train"]["seed"])

    requested_device = config["train"]["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA khong kha dung, chuyen sang CPU.")
    else:
        device = torch.device(requested_device)

    save_dir = config["project"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    resume_checkpoint_path = config["train"].get(
        "resume_checkpoint",
        os.path.join(save_dir, config["project"]["last_model_name"]),
    )

    if not os.path.exists(resume_checkpoint_path):
        raise FileNotFoundError(
            f"Khong tim thay checkpoint de resume: {resume_checkpoint_path}\n"
            f"Hay kiem tra train.resume_checkpoint trong {args.config}."
        )

    best_loss_model_path = os.path.join(
        save_dir,
        config["project"]["best_loss_model_name"],
    )

    last_model_path = os.path.join(
        save_dir,
        config["project"]["last_model_name"],
    )

    best_map_model_path = os.path.join(
        save_dir,
        config["project"]["best_map_model_name"],
    )

    image_size = config["data"]["image_size"]
    scales = config["data"]["scales"]
    anchors = config["anchors"]

    scaled_anchors = prepare_scaled_anchors(
        anchors=anchors,
        image_size=image_size,
        scales=scales,
        device=device,
    )

    augment_config = config.get("augment", {})
    train_augmenter = None

    if augment_config.get("enabled", False):
        if DetectionAugmenter is None:
            raise ImportError("augment.enabled=True but augmentations.py is not available.")
        train_augmenter = DetectionAugmenter(augment_config)

    imagenet_normalize = config.get("model", {}).get("imagenet_normalize", False)

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
        imagenet_normalize=imagenet_normalize,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(config=config, device=device)
    loss_fn = build_loss(config=config)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )

    use_amp = (
        config["train"].get("mixed_precision", False)
        and device.type == "cuda"
    )

    try:
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

    checkpoint = torch.load(
        resume_checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    load_optimizer_state = config["train"].get("load_optimizer_state", True)
    has_optimizer_state = "optimizer_state_dict" in checkpoint

    if load_optimizer_state and has_optimizer_state:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Loaded optimizer state from checkpoint.")
    else:
        print("Skipped optimizer state. Using a new optimizer.")
        set_optimizer_lr(optimizer, config["train"]["learning_rate"])

    scheduler = build_scheduler(config=config, optimizer=optimizer)
    has_scheduler_state = checkpoint.get("scheduler_state_dict", None) is not None

    if scheduler is not None:
        if has_scheduler_state and config["train"].get("load_scheduler_state", True):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("Loaded scheduler state from checkpoint.")
        else:
            # Old checkpoints may not contain scheduler_state_dict.
            # Reconstruct LR schedule from resumed epoch.
            if config["train"].get("rebuild_scheduler_from_epoch", True):
                set_optimizer_lr(optimizer, config["train"]["learning_rate"])
                completed_steps = compute_completed_scheduler_steps(
                    config=config,
                    resumed_epoch=int(checkpoint["epoch"]),
                )
                fast_forward_scheduler(scheduler, completed_steps)
                print(
                    "Rebuilt scheduler from epoch. "
                    f"completed_scheduler_steps={completed_steps}"
                )
            else:
                print("Scheduler created but not fast-forwarded.")

    if scaler is not None:
        scaler_state = checkpoint.get("scaler_state_dict", None)
        if scaler_state is not None and config["train"].get("load_scaler_state", True):
            scaler.load_state_dict(scaler_state)
            print("Loaded AMP scaler state from checkpoint.")

    resumed_epoch = int(checkpoint["epoch"])
    start_epoch = resumed_epoch + 1
    epochs = int(config["train"]["epochs"])

    best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))
    best_map50 = checkpoint.get("best_map50", checkpoint.get("map50", 0.0))
    if best_map50 is None:
        best_map50 = 0.0

    best_val_loss_at_best_map = checkpoint.get(
        "best_val_loss_at_best_map",
        checkpoint.get("val_loss", float("inf")),
    )

    map_eval_interval = config["train"]["map_eval_interval"]
    map_eval_start_epoch = config["train"].get("map_eval_start_epoch", 1)

    early_config = config.get("early_stopping", {})
    early_enabled = early_config.get("enabled", False)
    early_patience = early_config.get("patience", 3)
    early_min_delta = early_config.get("min_delta", 0.001)
    early_start_epoch = early_config.get("start_epoch", 0)
    early_bad_count = checkpoint.get("early_bad_count", 0)

    print("\nResume Kaggle training")
    print(f"Config       : {args.config}")
    print(f"Device       : {device}")
    print(f"Model        : {config.get('model', {}).get('name', 'yolov3_from_scratch')}")
    print(f"Freeze BB    : {config.get('model', {}).get('freeze_backbone', False)}")
    print(f"Checkpoint   : {resume_checkpoint_path}")
    print(f"Resumed epoch: {resumed_epoch}")
    print(f"Start epoch  : {start_epoch}")
    print(f"End epoch    : {epochs}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples  : {len(val_dataset)}")
    print(f"Save dir     : {save_dir}")
    print(f"Current LR   : {optimizer.param_groups[0]['lr']:.8f}")

    if start_epoch > epochs:
        print("Checkpoint epoch already >= configured epochs. Nothing to train.")
        return

    for epoch in range(start_epoch, epochs + 1):
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
            device=device,
        )

        should_eval_map = (
            epoch >= map_eval_start_epoch
            and (
                epoch % map_eval_interval == 0
                or epoch == epochs
            )
        )

        map50 = None
        micro_precision = None
        micro_recall = None

        if should_eval_map:
            predictions_json = build_val_predictions(
                model=model,
                config=config,
                device=device,
                conf_thresh=config["train"].get("map_conf_thresh", 0.25),
                iou_thresh=config["train"].get("map_nms_iou_thresh", 0.5),
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

        # Save last model every epoch so Kaggle interruption can be resumed.
        save_checkpoint_v2(
            save_path=last_model_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            map50=map50,
            config=config,
            best_val_loss=best_val_loss,
            best_map50=best_map50,
            best_val_loss_at_best_map=best_val_loss_at_best_map,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint_v2(
                save_path=best_loss_model_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                map50=map50,
                config=config,
                best_val_loss=best_val_loss,
                best_map50=best_map50,
                best_val_loss_at_best_map=best_val_loss_at_best_map,
            )

            print(
                f"Saved best loss model: {best_loss_model_path} "
                f"with best val_loss={best_val_loss:.4f}"
            )

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

                save_checkpoint_v2(
                    save_path=best_map_model_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=best_val_loss_at_best_map,
                    map50=best_map50,
                    config=config,
                    best_val_loss=best_val_loss,
                    best_map50=best_map50,
                    best_val_loss_at_best_map=best_val_loss_at_best_map,
                )

                print(
                    f"Saved best MAP model: {best_map_model_path} "
                    f"with best mAP@0.5={best_map50:.4f}"
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

    print("\nResume training completed.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val loss at best MAP@0.5: {best_val_loss_at_best_map:.4f}")
    print(f"Best MAP@0.5 score: {best_map50:.4f}")
    print(f"Best model saved at: {best_map_model_path}")


if __name__ == "__main__":
    main()
