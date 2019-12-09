import torch
import torch.nn as nn
from torchvision import models




def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

def convrelu_transpose(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding = 1, output_padding=padding, bias=True),
        nn.ReLU(inplace=True),
    )

class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        model = models.resnet18()
        model.load_state_dict(torch.load("resnet18-5c106cde.pth"))
        self.base_model = model
        
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):

        x_original = self.conv_original_size0(input)

        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

class ResNetUNetTranspose(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        model = models.resnet18()
        model.load_state_dict(torch.load("resnet18-5c106cde.pth"))
        self.base_model = model
        self.classification_model = model
        self.classification_model.fc = nn.Linear(512,2)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)
        self.reshape_channels = convrelu(512, 3, 1, 0)
        
        self.conv_transpose_3 = convrelu_transpose(512, 512, 3, 1)
        self.conv_transpose_2 = convrelu_transpose(512, 256, 3, 1)
        self.conv_transpose_1 = convrelu_transpose(256, 256, 3, 1)
        self.conv_transpose_0 = convrelu_transpose(256, 128, 3, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        #print("1", x_original.shape)
        x_original = self.conv_original_size1(x_original)
        #print("2", x_original.shape)
        #print("Input: ",input.shape)
        #print(self.layer0)
        layer0 = self.layer0(input)
        #print("Layer0: ",layer0.shape)
        #print("3", layer0.shape)
        layer0 = self.dropout(layer0)
        #print(self.layer1)
        layer1 = self.layer1(layer0)
        #print("Layer1: ",layer1.shape)
        #print("4", layer1.shape)
        layer1 = self.dropout(layer1)
        #print(self.layer2)
        layer2 = self.layer2(layer1)
        
        #print("Layer2: ",layer2.shape)
        layer2 = self.dropout(layer2)
        #print("5", layer2.shape)
        #print(self.layer3)
        layer3 = self.layer3(layer2)
        #print("Layer3: ",layer3.shape)
        #print("6",layer3.shape)
        layer3 = self.dropout(layer3)
        #print(self.layer4)
        layer4 = self.layer4(layer3)
        #print("Layer4: ",layer4.shape)
        #print("7", layer4.shape)
        layer4 = self.dropout(layer4)
        #print(self.layer4_1x1)
        layer4 = self.layer4_1x1(layer4)
        #print("8", layer4.shape)
        #print("Pre classification arm: ", layer4.shape)
        #Classification arm
        layer5 = self.reshape_channels(layer4)
        classification = self.classification_model(layer5)
        #print("Layer4 in", layer4.shape)
        #print("Layer4", layer4.shape)
        #print(self.conv_transpose_3)
        x = self.conv_transpose_3(layer4)
        #print("x", x.shape)
        #print(self.conv_transpose_2)
        #print("9", x.shape)
        #print("Layer3 out", x.shape)
        x = self.conv_transpose_2(x)
        #print("x", x.shape)
        #print(self.conv_transpose_1)
        #print("10", x.shape)
        #print("Layer2 out", x.shape)
        x = self.conv_transpose_1(x)
        #print("x", x.shape)
        #print(self.conv_transpose_0)
        #print("11", x.shape)
        #print("Layer1 out", x.shape)
        x = self.conv_transpose_0(x)
        #print("Layer 0 out", x.shape)
        #print("x", x.shape)

        x = self.upsample(x)
        #print("x", x.shape)
        #print("x_original",x_original.shape)
        x = torch.cat([x, x_original], dim=1)
        #print("x", x.shape)
        #print(self.conv_original_size2)
        x = self.conv_original_size2(x)
        #print("x", x.shape)
        #print(self.conv_last)
        out = self.conv_last(x)
        #print("out", classification.shape)
        
        return out, classification

    
class ResNetUNetSegmentation(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        model = models.resnet18()
        model.load_state_dict(torch.load("resnet18-5c106cde.pth"))
        self.base_model = model
        #self.classification_model = model
        #self.classification_model.fc = nn.Linear(512,1)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)
        self.reshape_channels = convrelu(512, 3, 1, 0)
        
        self.conv_transpose_3 = convrelu_transpose(512, 512, 3, 1)
        self.conv_transpose_2 = convrelu_transpose(512, 256, 3, 1)
        self.conv_transpose_1 = convrelu_transpose(256, 256, 3, 1)
        self.conv_transpose_0 = convrelu_transpose(256, 128, 3, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        #print("1", x_original.shape)
        x_original = self.conv_original_size1(x_original)
        #print("2", x_original.shape)
        layer0 = self.layer0(input)
        #print("3", layer0.shape)
        layer1 = self.layer1(layer0)
        #print("4", layer1.shape)
        layer2 = self.layer2(layer1)
        #print("5", layer2.shape)
        layer3 = self.layer3(layer2)
        #print("6",layer3.shape)
        layer4 = self.layer4(layer3)
        #print("7", layer4.shape)

        layer4 = self.layer4_1x1(layer4)
        #print("8", layer4.shape)
        
        #Classification arm
        #layer5 = self.reshape_channels(layer4)
        #classification = self.classification_model(layer5)
        #print("Layer4 in", layer4.shape)
        x = self.conv_transpose_3(layer4)
        #print("9", x.shape)
        #print("Layer3 out", x.shape)
        x = self.conv_transpose_2(x)
        #print("10", x.shape)
        #print("Layer2 out", x.shape)
        x = self.conv_transpose_1(x)
        #print("11", x.shape)
        #print("Layer1 out", x.shape)
        x = self.conv_transpose_0(x)
        #print("Layer 0 out", x.shape)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out, None