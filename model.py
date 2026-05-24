import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision
from tqdm import tqdm

#CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
seed = 17
torch.manual_seed(17)
print(device)

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=not bn_act,
            **kwargs
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)

        self.use_bn_act = bn_act
    
    def forward(self, x):
        x = self.conv(x)

        if self.use_bn_act:
            x = self.bn(x)
            x = self.leaky(x)
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats = 1):
        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(num_repeats):
            self.layers.append(
                nn.Sequential(
                    CNNBlock(
                        in_channels=channels,
                        out_channels=channels // 2,
                        kernel_size=1,
                        stride=1,
                        padding=0
                    ),
                    CNNBlock(
                        in_channels=channels // 2,
                        out_channels=channels,
                        kernel_size=3,
                        stride=1,
                        padding=1 # Tại sao ở đây padding = 1 ?
                    )
                )
            )
        
        self.use_residual = use_residual
        self.num_repeats = num_repeats
    
    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        
        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.num_classes = num_classes

        # Chưa hiểu hàm này
        self.pred = nn.Sequential(
            CNNBlock(
                in_channels=in_channels,
                out_channels=2*in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            CNNBlock(
                in_channels=2*in_channels,
                out_channels=3*(num_classes + 5),
                bn_act=False,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
    def forward(self, x):
        batch_size = x.shape[0]

        x = self.pred(x)

        # Before reshape:
        # [B, 3 * (num_classes + 5), S, S]
        #
        # After reshape:
        # [B, 3, S, S, num_classes + 5]
        x = x.reshape(
            batch_size,
            3,
            self.num_classes + 5,
            x.shape[2],
            x.shape[3]
        )

        x = x.permute(0, 1, 3, 4, 2)

        return x

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()

        self.num_classes = num_classes

        self.config = [
            # Darket53 backbone
            # (out_channels/filters, kernel_size, stride)
            (32, 3, 1),
            (64, 3, 2),
            ["B", 1],

            (128, 3, 2),
            ["B", 2],

            (256, 3, 2),
            ["B", 8],

            (512, 3, 2),
            ["B", 8],

            (1024, 3, 2),
            ["B", 4],

            # Detection block for large objects
            (512, 1, 1),
            (1024, 3, 1),
            (512, 1, 1),
            (1024, 3, 1),
            (512, 1, 1),
            "S",

            # Upsample + concat with 26x26 feature map
            (256, 1, 1),
            "U",

            # Detection block for medium objects
            (256, 1, 1),
            (512, 3, 1),
            (256, 1, 1),
            (512, 3, 1),
            (256, 1, 1),
            "S",

            # Upsample + concat with 52x52 feature map
            (128, 1, 1),
            "U",

            # Detection block for small objects
            (128, 1, 1),
            (256, 3, 1),
            (128, 1, 1),
            (256, 3, 1),
            (128, 1, 1),
            "S",
        ]

        self.layers = self._create_conv_layers(in_channels)
    
    def _create_conv_layers(self, in_channels):
        layers = nn.ModuleList()

        for module in self.config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module

                padding = 1 if kernel_size == 3 else 0

                layers.append(
                    CNNBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding
                    )
                )

                in_channels = out_channels
            
            elif isinstance(module, list):
                num_repeats = module[1]

                layers.append(
                    ResidualBlock(
                        channels=in_channels,
                        use_residual=True,
                        num_repeats=num_repeats
                    )
                )
            
            elif module == "S":
                layers.append(
                    ResidualBlock(
                        channels=in_channels,
                        use_residual=False,
                        num_repeats=1
                    )
                )

                layers.append(
                    CNNBlock(
                        in_channels=in_channels,
                        out_channels=in_channels // 2,
                        kernel_size=1,
                        stride=1,
                        padding=0
                    )
                )

                layers.append(
                    ScalePrediction(
                        in_channels=in_channels // 2,
                        num_classes=self.num_classes
                    )
                )

                in_channels = in_channels // 2
            
            elif module == "U":
                layers.append(
                    nn.Upsample(
                        scale_factor=2,
                        mode="nearest"
                    )
                )

                # Sau upsample, ta sẽ concat với feature map từ backbone.
                # Ví dụ: 26×26×256 concat 26×26×512 = 26×26×768.
                # 768 = 256 * 3.
                in_channels = in_channels * 3

        return layers

    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            
            x = layer(x)

            # Lưu feature map 52×52 và 26×26 từ Darknet-53
            # Hai feature map này dùng để concat sau upsample
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            
            elif isinstance(layer, nn.Upsample):
                route_connection = route_connections.pop()
                x = torch.cat([x, route_connection], dim = 1)
        
        return outputs

if __name__ == "__main__":
    num_classes = 5
    image_size = 416

    model = YOLOv3(in_channels=3, num_classes=num_classes)

    x = torch.randn(2, 3, image_size, image_size)

    outputs = model(x)

    for out in outputs:
        print(out.shape)

