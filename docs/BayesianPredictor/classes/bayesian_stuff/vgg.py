import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F

__all__ = ['vgg19']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGG(nn.Module):
    def __init__(self, features, include_capacity=False):
        super(VGG, self).__init__()
        self.features = features
        self.include_capacity = include_capacity
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
        
        # Capacity prediction head: outputs a single scalar value (0-100)
        if self.include_capacity:
            self.capacity_layer = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid()  # Output between 0 and 1 (representing 0-100%)
            )

    def forward(self, x):
        x = self.features(x)
        
        # Density map prediction
        density = F.upsample_bilinear(x, scale_factor=2)
        density = self.reg_layer(density)
        density = torch.abs(density)
        
        # Capacity prediction (optional)
        if self.include_capacity:
            capacity = self.capacity_layer(x)
            capacity = capacity * 100  # Scale from [0, 1] to [0, 100]
            return density, capacity
        else:
            return density


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19(include_capacity=False):
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
        
    Args:
        include_capacity: If True, the model will also output beach capacity predictions
    """
    model = VGG(make_layers(cfg['E']), include_capacity=include_capacity)
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model

