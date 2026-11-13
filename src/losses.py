import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits).view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return bce_loss + (1 - dice.mean())


@torch.no_grad()
def dice_iou_per_image(logits, targets, threshold=0.5, smooth=1e-6):
    """Return (dice, iou), each a tensor with one value per image."""
    preds = (torch.sigmoid(logits) > threshold).float().view(logits.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    dice = (2 * intersection + smooth) / (preds.sum(dim=1) + targets.sum(dim=1) + smooth)
    iou = (intersection + smooth) / (union + smooth)
    return dice, iou


@torch.no_grad()
def dice_iou(logits, targets, threshold=0.5):
    """Batch-average Dice and IoU as plain Python floats."""
    dice, iou = dice_iou_per_image(logits, targets, threshold)
    return dice.mean().item(), iou.mean().item()