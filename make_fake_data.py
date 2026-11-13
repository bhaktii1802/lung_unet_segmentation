import os

import cv2
import numpy as np

rng = np.random.default_rng(0)
for split, n in [("train", 24), ("val", 8), ("test", 8)]:
    for sub in ("image", "mask"):
        os.makedirs(f"data_fake/ChestXray/{split}/{sub}", exist_ok=True)
    for i in range(n):
        mask = np.zeros((128, 128), np.uint8)
        cx, cy = rng.integers(40, 88, 2)
        a, b = rng.integers(15, 35, 2)
        cv2.ellipse(mask, (int(cx), int(cy)), (int(a), int(b)), 0, 0, 360, 255, -1)
        noise = rng.integers(0, 60, (128, 128)).astype(np.uint8)
        image = (mask // 2 + noise).astype(np.uint8)
        cv2.imwrite(f"data_fake/ChestXray/{split}/image/cxrimage_{i}.png", image)
        cv2.imwrite(f"data_fake/ChestXray/{split}/mask/cxrmask_{i}.jpeg", mask)
print("fake data created")