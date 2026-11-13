import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import LungDataset
from losses import dice_iou_per_image
from model import UNet
from utils import get_device


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained U-Net")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out-dir", default="outputs/eval")
    p.add_argument("--img-size", type=int, default=None, help="default: read from checkpoint")
    p.add_argument("--base", type=int, default=None, help="default: read from checkpoint")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--n-worst", type=int, default=4)
    return p.parse_args()


def load_checkpoint(path, device):
    """Accepts both our dict format and a plain state_dict (like your old best_unet_full.pth)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"], ckpt.get("config", {})
    return ckpt, {}


def main():
    args = parse_args()
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    state, cfg = load_checkpoint(args.checkpoint, device)
    img_size = args.img_size or cfg.get("img_size", 256)
    base = args.base or cfg.get("base", 32)

    model = UNet(in_ch=1, out_ch=1, base=base).to(device)
    model.load_state_dict(state)
    model.eval()

    ds = LungDataset(os.path.join(args.data_dir, args.split), img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device == "cuda")

    # Score every image
    dices, ious = [], []
    with torch.no_grad():
        for images, masks in loader:
            logits = model(images.to(device))
            d, i = dice_iou_per_image(logits.cpu(), masks)
            dices.append(d)
            ious.append(i)
    dices, ious = torch.cat(dices), torch.cat(ious)

    print(f"{args.split.upper()}  Dice {dices.mean():.4f} | IoU {ious.mean():.4f} | "
          f"min Dice {dices.min():.4f} | images with Dice < 0.9: "
          f"{(dices < 0.9).sum().item()} of {len(dices)}")

    # Save per-image scores with real filenames
    csv_path = os.path.join(args.out_dir, f"{args.split}_scores.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "filename", "dice", "iou"])
        for idx, name in enumerate(ds.filenames):
            w.writerow([idx, name, f"{dices[idx]:.4f}", f"{ious[idx]:.4f}"])

    # Plot the worst images
    k = min(args.n_worst, len(ds))
    worst = dices.argsort()[:k].tolist()
    fig, ax = plt.subplots(k, 3, figsize=(9, 3 * k), squeeze=False)
    for r, idx in enumerate(worst):
        image, mask = ds[idx]
        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits) > 0.5).float().cpu()[0, 0]
        print(f"worst #{r + 1}: index {idx} {ds.filenames[idx]} Dice {dices[idx]:.3f}")
        ax[r, 0].imshow(image[0], cmap="gray"); ax[r, 0].set_title(ds.filenames[idx], fontsize=8)
        ax[r, 1].imshow(mask[0], cmap="gray"); ax[r, 1].set_title("true mask")
        ax[r, 2].imshow(pred, cmap="gray"); ax[r, 2].set_title(f"pred (Dice {dices[idx]:.3f})")
        for a in ax[r]:
            a.axis("off")
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, f"{args.split}_worst_cases.png")
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"Saved {csv_path} and {fig_path}")


if __name__ == "__main__":
    main()