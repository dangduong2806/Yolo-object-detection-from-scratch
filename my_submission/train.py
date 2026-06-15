import argparse
import os
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "utils"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--val_image_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    return parser.parse_args()


def infer_root_dir(image_dir):
    image_dir = Path(image_dir)
    # Expected public layout: public/train/images and public/val/images.
    if image_dir.name == "images" and image_dir.parent.name in {"train", "val"}:
        return image_dir.parent.parent
    return image_dir


def load_base_config():
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_base_config()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config["project"]["save_dir"] = str(checkpoint_dir)
    config["project"]["best_map_model_name"] = "best.pth"
    config["project"]["best_loss_model_name"] = "best_loss.pth"
    config["project"]["last_model_name"] = "last.pth"

    config["data"]["root_dir"] = str(infer_root_dir(args.image_dir))
    config["data"]["train_json"] = args.train_data
    config["data"]["val_json"] = args.val_data

    config["train"]["epochs"] = args.epochs
    config["train"]["batch_size"] = args.batch_size
    config["train"]["learning_rate"] = args.lr
    config["train"]["device"] = args.device
    config["train"]["num_workers"] = args.num_workers
    config["train"]["resume_checkpoint"] = args.resume_checkpoint
    config["train"]["load_optimizer_state"] = False

    generated_config = ROOT / "_generated_train_config.yaml"
    generated_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    os.chdir(ROOT)
    import train_core  # noqa: E402

    original_load_config = train_core.load_config

    def load_generated_config(_config_path):
        return original_load_config(str(generated_config))

    train_core.load_config = load_generated_config
    train_core.main()


if __name__ == "__main__":
    main()
