import torch
import segmentation_models_pytorch as smp
from torch import nn
from segmentation_models_pytorch.losses import DiceLoss


class SegmentationModel(nn.Module):
    """Segmentation model using a Unet architecture."""

    def __init__(self, encoder_name='timm-efficientnet-b0', weights='imagenet'):
        super(SegmentationModel, self).__init__()
        self.arc = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, images, masks=None):
        logits = self.arc(images)
        if masks is not None:
            loss1 = DiceLoss(mode='binary')(logits, masks)
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            return logits, loss1 + loss2
        return logits
