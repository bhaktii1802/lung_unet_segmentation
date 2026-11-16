import sys
sys.path.insert(0, "src")

import matplotlib.pyplot as plt
from dataset import LungDataset

ds = LungDataset("data_fake/ChestXray/train", img_size=128, augment=True)
fig, ax = plt.subplots(4, 4, figsize=(10, 10))
for r in range(4):
    for c in range(2):
        image, mask = ds[r]
        ax[r, c * 2].imshow(image[0], cmap="gray")
        ax[r, c * 2 + 1].imshow(mask[0], cmap="gray")
        ax[r, c * 2].axis("off"); ax[r, c * 2 + 1].axis("off")
plt.tight_layout()
plt.savefig("check_aug.png")
print("saved check_aug.png")