# Chạy train trên 1 batch cố định để kiểm tra vấn đề thiếu data augmentation ?

import argparse
import copy

import torch
from torch.utils.data import DataLoader

from dataset_kaggle import YOLODataset
from loss import YOLOv3Loss
from model import YOLOv3
from model_resnet_yolov3 import ResNet50_Weights, ResNet50YOLOv3
from train import (
    compute_yolo_loss,
    load_config,
    move_targets_to_device,
    prepare_scaled_anchors,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overfit one fixed batch to verify the YOLO training pipeline."
    )

    parser.add_argument("--config", type=str, default="config_kaggle.yaml")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--use-val",
        action="store_true",
        help="Use one validation batch instead of one training batch.",
    )

    return parser.parse_args()


def clone_batch(images, targets):
    images = images.clone()
    targets = tuple(target.clone() for target in targets)
    return images, targets


def main():
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["train"]["seed"])

    requested_device = args.device or config["train"]["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")
    else:
        device = torch.device(requested_device)

    batch_size = args.batch_size or config["train"]["batch_size"]
    image_size = config["data"]["image_size"]
    scales = config["data"]["scales"]
    anchors = config["anchors"]

    json_path = (
        config["data"]["val_json"]
        if args.use_val
        else config["data"]["train_json"]
    )

    imagenet_normalize=config.get("model", {}).get("imagenet_normalize", False),

    dataset = YOLODataset(
        json_path=json_path,
        root_dir=config["data"]["root_dir"],
        anchors=anchors,
        image_size=image_size,
        scales=scales,
        ignore_iou_thresh=config["data"]["ignore_iou_thresh"],
        use_letterbox=config["data"]["use_letterbox"],
        imagenet_normalize=imagenet_normalize,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    images, targets = clone_batch(*next(iter(loader)))

    positive_counts = [
        int((target[..., 0] == 1).sum().item())
        for target in targets
    ]
    ignore_counts = [
        int((target[..., 0] == -1).sum().item())
        for target in targets
    ]

    print("Single-batch overfit test")
    print(f"Dataset      : {'val' if args.use_val else 'train'}")
    print(f"Device       : {device}")
    print(f"Batch size   : {batch_size}")
    print(f"Steps        : {args.steps}")
    print(f"LR           : {args.lr}")
    print(f"Weight decay : {args.weight_decay}")
    print(f"Positive 13/26/52: {positive_counts}")
    print(f"Ignore   13/26/52: {ignore_counts}")

    images = images.to(device)
    targets = move_targets_to_device(targets=targets, device=device)

    scaled_anchors = prepare_scaled_anchors(
        anchors=anchors,
        image_size=image_size,
        scales=scales,
        device=device,
    )

    # model = YOLOv3(
    #     in_channels=3,
    #     num_classes=config["data"]["num_classes"],
    # ).to(device)

    model_name = config.get("model", {}).get("name", "yolov3_from_scratch")

    if model_name == "resnet50_yolov3":
        model = ResNet50YOLOv3(
            in_channels=3,
            num_classes=config["data"]["num_classes"],
            pretrained=False,
            freeze_backbone=False,
        ).to(device)
    else:
        model = YOLOv3(
            in_channels=3,
            num_classes=config["data"]["num_classes"],
        ).to(device)

    # loss_fn = YOLOv3Loss(
    #     num_classes=config["data"]["num_classes"],
    #     lambda_box=config["loss"]["lambda_box"],
    #     lambda_obj=config["loss"]["lambda_obj"],
    #     lambda_noobj=config["loss"]["lambda_noobj"],
    #     lambda_class=config["loss"]["lambda_class"],
    # )
    loss_fn = YOLOv3Loss(
        num_classes=config["data"]["num_classes"],
        lambda_box=config["loss"]["lambda_box"],
        lambda_obj=config["loss"]["lambda_obj"],
        lambda_noobj=config["loss"]["lambda_noobj"],
        lambda_class=config["loss"]["lambda_class"],
        box_loss_type=config["loss"].get("box_loss_type", "mse"),
        obj_label_smoothing=config["loss"].get("obj_label_smoothing", 0.0),
        class_label_smoothing=config["loss"].get("class_label_smoothing", 0.0),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    first_logs = None
    last_logs = None

    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss, logs = compute_yolo_loss(
            outputs=outputs,
            targets=targets,
            scaled_anchors=scaled_anchors,
            loss_fn=loss_fn,
        )

        loss.backward()

        gradient_clip_norm = config["train"].get("gradient_clip_norm")
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip_norm,
            )

        optimizer.step()

        last_logs = copy.deepcopy(logs)
        if first_logs is None:
            first_logs = copy.deepcopy(logs)

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                f"step={step:04d} "
                f"total={loss.item():.6f} "
                f"box={logs['box_loss']:.6f} "
                f"obj={logs['obj_loss']:.6f} "
                f"noobj={logs['noobj_loss']:.6f} "
                f"class={logs['class_loss']:.6f}"
            )

    print("\nSummary")
    for key in ["box_loss", "obj_loss", "noobj_loss", "class_loss"]:
        start = first_logs[key]
        end = last_logs[key]
        ratio = end / max(start, 1e-12)
        print(f"{key}: {start:.6f} -> {end:.6f} ratio={ratio:.4f}")

    print(
        "\nExpected: on one fixed batch, box/obj/class losses should drop "
        "strongly. If they do not, inspect target assignment, loss, model "
        "output order, or anchor scaling before training full runs."
    )


if __name__ == "__main__":
    main()
