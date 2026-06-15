import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            bias=not bn_act,
            **kwargs
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.use_bn_act = bn_act

    def forward(self, x):
        x = self.conv(x)

        if self.use_bn_act:
            x = self.bn(x)
            x = self.act(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.pred = nn.Sequential(
            CNNBlock(
                in_channels=in_channels,
                out_channels=2 * in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            CNNBlock(
                in_channels=2 * in_channels,
                out_channels=3 * (num_classes + 5),
                bn_act=False,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.pred(x)

        x = x.reshape(
            batch_size,
            3,
            self.num_classes + 5,
            x.shape[2],
            x.shape[3],
        )

        x = x.permute(0, 1, 3, 4, 2)

        return x


class ResNet50YOLOv3(nn.Module):
    """
    ResNet50 pretrained backbone + YOLOv3-like neck/head.

    Output giữ nguyên format cũ:
        outputs[0]: [B, 3, 13, 13, 5 + C]
        outputs[1]: [B, 3, 26, 26, 5 + C]
        outputs[2]: [B, 3, 52, 52, 5 + C]
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=5,
        pretrained=True,
        freeze_backbone=False,
    ):
        super().__init__()

        if in_channels != 3:
            raise ValueError("ResNet pretrained backbone chỉ hỗ trợ input RGB 3 channels.")

        self.num_classes = num_classes

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        # Input 416x416
        # conv1 + maxpool -> 104x104
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        # layer1: 104x104, channels 256
        # layer2: 52x52, channels 512
        # layer3: 26x26, channels 1024
        # layer4: 13x13, channels 2048
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if freeze_backbone:
            for param in self.stem.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters():
                param.requires_grad = False
            for param in self.layer3.parameters():
                param.requires_grad = False
            for param in self.layer4.parameters():
                param.requires_grad = False

        # 13x13 large-object path
        self.c5_reduce = CNNBlock(2048, 512, kernel_size=1, stride=1, padding=0)

        self.p5_head = nn.Sequential(
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
        )

        self.pred13 = ScalePrediction(512, num_classes)

        # 26x26 medium-object path
        self.p5_up = nn.Sequential(
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        self.c4_reduce = CNNBlock(1024, 256, kernel_size=1, stride=1, padding=0)

        self.p4_head = nn.Sequential(
            CNNBlock(512, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
        )

        self.pred26 = ScalePrediction(256, num_classes)

        # 52x52 small-object path
        self.p4_up = nn.Sequential(
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        self.c3_reduce = CNNBlock(512, 128, kernel_size=1, stride=1, padding=0)

        self.p3_head = nn.Sequential(
            CNNBlock(256, 256, kernel_size=3, stride=1, padding=1),
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0),
            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1),
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0),
        )

        self.pred52 = ScalePrediction(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)  # 52x52
        c4 = self.layer3(c3)  # 26x26
        c5 = self.layer4(c4)  # 13x13

        p5 = self.c5_reduce(c5)
        p5 = self.p5_head(p5)
        out13 = self.pred13(p5)

        p5_up = self.p5_up(p5)
        c4_reduced = self.c4_reduce(c4)
        p4 = torch.cat([p5_up, c4_reduced], dim=1)
        p4 = self.p4_head(p4)
        out26 = self.pred26(p4)

        p4_up = self.p4_up(p4)
        c3_reduced = self.c3_reduce(c3)
        p3 = torch.cat([p4_up, c3_reduced], dim=1)
        p3 = self.p3_head(p3)
        out52 = self.pred52(p3)

        return [out13, out26, out52]


if __name__ == "__main__":
    model = ResNet50YOLOv3(num_classes=5, pretrained=False)

    x = torch.randn(2, 3, 416, 416)
    outputs = model(x)

    for out in outputs:
        print(out.shape)