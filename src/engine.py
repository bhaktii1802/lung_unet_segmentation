import torch

from losses import dice_iou


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Train for one epoch if an optimizer is given, otherwise just evaluate."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, total_dice, total_iou, n = 0.0, 0.0, 0.0, 0

    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            if training:
                optimizer.zero_grad()   # clear old gradients
                loss.backward()         # compute new gradients
                optimizer.step()        # update the weights

            bs = images.size(0)
            d, i = dice_iou(logits, masks)
            total_loss += loss.item() * bs
            total_dice += d * bs
            total_iou += i * bs
            n += bs

    return total_loss / n, total_dice / n, total_iou / n