from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


base = Path(__file__).parent
output = base / "mnist_examples"
output.mkdir(exist_ok=True)
saved = set()

# Leer por bloques para consumir poca memoria.
for chunk in pd.read_csv(base / "mnist_small.csv", chunksize=500):
    columns = chunk.filter(regex=r"^pixel\d+$").columns

    if len(columns) != 784:
        raise ValueError("Expected 784 pixel columns.")

    for digit in range(10):
        if digit in saved:
            continue

        rows = chunk[chunk["Target"] == digit]
        if rows.empty:
            continue

        # Conservar el orden de píxeles usado durante el entrenamiento.
        pixels = rows[columns].iloc[0].to_numpy(dtype="float32")
        pixels = np.clip(pixels, 0, 255).astype("uint8")

        Image.fromarray(pixels.reshape(28, 28)).save(
            output / f"digit_{digit}.png"
        )
        saved.add(digit)

    if len(saved) == 10:
        break

print(f"Saved {len(saved)} images in mnist_examples.")