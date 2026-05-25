import torch

IMAGE_SIZE = 416
NUM_CLASSES = 5
SCALES = [13, 26, 52]

ANCHORS = torch.tensor([
    [[116, 90], [156, 198], [373, 326]], # scale 13x13
    [[30, 61], [62, 45],   [59, 119]], # scale 26x26
    [[10, 13],  [16, 30],   [33, 23]],    # scale 52×52
], dtype=torch.float32)

