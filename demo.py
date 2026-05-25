import yaml
import torch
from torch.utils.data import DataLoader

from model import YOLOv3
from dataset import YOLODataset
from loss import YOLOv3Loss


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_scaled_anchors(anchors, image_size, scales, device):
    anchors = torch.tensor(anchors, dtype=torch.float32)

    # anchors pixel -> anchors normalized
    anchors_normalized = anchors / image_size

    # anchors normalized -> anchors theo đơn vị grid cell
    scales_tensor = torch.tensor(
        scales,
        dtype=torch.float32
    ).reshape(3, 1, 1)

    scaled_anchors = anchors_normalized * scales_tensor

    return scaled_anchors.to(device)


def main():
    config = load_config("config.yaml")

    image_size = config["data"]["image_size"]
    num_classes = config["data"]["num_classes"]
    scales = config["data"]["scales"]
    anchors = config["anchors"]

    device_name = config["train"]["device"]

    if device_name == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA không khả dụng, dùng CPU.")
    else:
        device = torch.device(device_name)

    print("Device:", device)

    dataset = YOLODataset(
        json_path=config["data"]["train_json"],
        root_dir=config["data"]["root_dir"],
        anchors=anchors,
        image_size=image_size,
        scales=scales,
        ignore_iou_thresh=config["data"]["ignore_iou_thresh"],
        use_letterbox=config["data"]["use_letterbox"],
    )

    train_loader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    model = YOLOv3(
        in_channels=3,
        num_classes=num_classes
    ).to(device)

    loss_fn = YOLOv3Loss(
        num_classes=num_classes,
        lambda_box=config["loss"]["lambda_box"],
        lambda_obj=config["loss"]["lambda_obj"],
        lambda_noobj=config["loss"]["lambda_noobj"],
        lambda_class=config["loss"]["lambda_class"],
    )

    scaled_anchors = prepare_scaled_anchors(
        anchors=anchors,
        image_size=image_size,
        scales=scales,
        device=device
    )

    images, targets = next(iter(train_loader))

    print("\n===== DATASET OUTPUT =====")
    print("images.shape:", images.shape)
    print("targets[0].shape:", targets[0].shape)
    print("targets[1].shape:", targets[1].shape)
    print("targets[2].shape:", targets[2].shape)

    for scale_idx, target in enumerate(targets):
        positive = (target[..., 0] == 1).sum().item()
        ignore = (target[..., 0] == -1).sum().item()
        noobj = (target[..., 0] == 0).sum().item()

        print(
            f"Scale {scales[scale_idx]}x{scales[scale_idx]} | "
            f"positive={positive}, ignore={ignore}, noobj={noobj}"
        )

    images = images.to(device)
    targets = [target.to(device) for target in targets]

    model.eval()

    with torch.no_grad():
        outputs = model(images)

    print("\n===== MODEL OUTPUT =====")
    print("outputs[0].shape:", outputs[0].shape)
    print("outputs[1].shape:", outputs[1].shape)
    print("outputs[2].shape:", outputs[2].shape)

    print("\n===== SCALED ANCHORS =====")
    print("scaled_anchors.shape:", scaled_anchors.shape)
    print("scaled_anchors[0]:", scaled_anchors[0])
    print("scaled_anchors[1]:", scaled_anchors[1])
    print("scaled_anchors[2]:", scaled_anchors[2])

    print("\n===== LOSS CHECK =====")

    total_loss = 0.0

    for scale_idx in range(3):
        scale_loss, loss_items = loss_fn(
            outputs[scale_idx],
            targets[scale_idx],
            scaled_anchors[scale_idx]
        )

        total_loss += scale_loss

        print(f"\nScale {scales[scale_idx]}x{scales[scale_idx]}")
        print("scale_loss:", scale_loss.item())
        print("box_loss:", loss_items["box_loss"].item())
        print("obj_loss:", loss_items["obj_loss"].item())
        print("noobj_loss:", loss_items["noobj_loss"].item())
        print("class_loss:", loss_items["class_loss"].item())

    print("\nTotal loss:", total_loss.item())

    print("\nDemo chạy thành công.")


if __name__ == "__main__":
    main()