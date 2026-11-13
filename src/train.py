import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # draw plots to files, no window needed
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from dataset import LungDataset
from engine import run_epoch
from losses import BCEDiceLoss
from model import UNet
from utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train a U-Net for lung segmentation")
    p.add_argument("--data-dir", required=True, help="folder that contains train/ val/ test/")
    p.add_argument("--out-dir", default="outputs/run", help="where to save results")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--base", type=int, default=32, help="base number of U-Net channels")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--subset", type=int, default=0, help="use only the first N images (quick tests)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def save_curves(history, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history["train_dice"], label="train")
    ax[1].plot(history["val_dice"], label="val")
    ax[1].set_title("Dice"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device: {device}")
    print(f"settings: {vars(args)}")

    # Data
    train_ds = LungDataset(os.path.join(args.data_dir, "train"), args.img_size)
    val_ds = LungDataset(os.path.join(args.data_dir, "val"), args.img_size)
    if args.subset:
        train_ds = Subset(train_ds, range(min(args.subset, len(train_ds))))
        val_ds = Subset(val_ds, range(min(args.subset, len(val_ds))))

    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    # Model, loss, optimizer
    model = UNet(in_ch=1, out_ch=1, base=args.base).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    config = {"img_size": args.img_size, "base": args.base}
    ckpt_path = os.path.join(args.out_dir, "best.pth")
    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    best_dice = 0.0

    # Training loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_dice, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_dice, va_iou = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_dice)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_dice"].append(tr_dice)
        history["val_dice"].append(va_dice)
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        saved = ""
        if va_dice > best_dice:
            best_dice = va_dice
            torch.save({"model": model.state_dict(), "config": config}, ckpt_path)
            saved = "  <- saved"

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} dice {tr_dice:.4f} | "
              f"val loss {va_loss:.4f} dice {va_dice:.4f} iou {va_iou:.4f} | "
              f"lr {lr:.1e} | {time.time() - t0:.0f}s{saved}")

    save_curves(history, os.path.join(args.out_dir, "curves.png"))
    print(f"Done. Best val Dice {best_dice:.4f}. Files saved in {args.out_dir}/")


if __name__ == "__main__":
    main()