import torch
import torch.nn as nn
from torchvision import models


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetPatch(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet50()
        model.load_state_dict(torch.load("resnet50-19c8e357.pth"))
        self.base_model = model

        self.base_layers = list(self.base_model.children())
        # print(self.base_layers[:3])
        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        # print("3 to 5")
        # print(self.base_layers[3:5])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        # self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 512, x.H/8, x.W/8)
        # self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 1024, x.H/16, x.W/16)
        # self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 2048, x.H/32, x.W/32)
        self.patch_out = convrelu(2048, 1, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        # self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        # print(x_original.shape)
        layer0 = self.layer0(input)
        # print("Layer 0", layer0.shape)
        layer1 = self.layer1(layer0)
        # print("Layer 1", layer1.shape)
        layer2 = self.layer2(layer1)
        # print("Layer 2", layer2.shape)
        layer3 = self.layer3(layer2)
        # print("Layer 3", layer3.shape)
        layer4 = self.layer4(layer3)
        # print("Layer 4", layer4.shape)
        # layer4 = self.layer4_1x1(layer4)
        out = self.patch_out(layer4)
        # print(out.shape)

        return out