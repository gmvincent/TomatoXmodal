import torch
import torch.nn as nn
from torchvision import models

class EfficientNetStudent(nn.Module):
    def __init__(self, model_name="efficientnet_b3", num_classes=10, pretrained=True, input_channels=3):
        super().__init__()
        # load base EfficientNet
        model_func = getattr(models, model_name)
        self.backbone = model_func(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)

        # adjust input channels
        if input_channels != 3:
            old_conv = self.backbone.features[0][0]
            new_conv = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                if input_channels > 3:
                    new_conv.weight[:, 3:] = old_conv.weight[:, :1].repeat(1, input_channels - 3, 1, 1)
            self.backbone.features[0][0] = new_conv

        # replace classifier
        self.num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(self.num_features, num_classes)

    def forward(self, x, return_features=False):
        """
        x: input image tensor
        return_features: if True, return feature vector before classifier
        """
        features = self.backbone.features(x)         # CNN features
        features = self.backbone.avgpool(features)  # global avg pool
        features = torch.flatten(features, 1)       # flatten to (B, num_features)

        logits = self.backbone.classifier(features)

        if return_features:
            return features, logits
        else:
            return logits
