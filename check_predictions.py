from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from image_utils import prepare_digit


base = Path(__file__).parent
model = tf.keras.models.load_model(
    base / "mnist_model.keras", compile=False
)

examples = {}

# Obtener los mismos primeros ejemplos usados al exportar las imágenes.
for chunk in pd.read_csv(base / "mnist_small.csv", chunksize=500):
    columns = chunk.filter(regex=r"^pixel\d+$").columns

    if len(columns) != 784:
        raise ValueError("Expected 784 pixel columns.")

    for digit in range(10):
        if digit in examples:
            continue

        rows = chunk[chunk["Target"] == digit]
        if not rows.empty:
            examples[digit] = rows[columns].iloc[0].to_numpy(
                dtype="float32"
            )

    if len(examples) == 10:
        break

results = []

for digit, pixels in sorted(examples.items()):
    # Predecir directamente desde el CSV, sin modificar la imagen.
    original = pixels.reshape(1, 784) / 255.0

    # Aplicar el mismo procesamiento utilizado en Streamlit.
    image = Image.fromarray(
        np.clip(pixels, 0, 255).astype("uint8").reshape(28, 28)
    )
    prepared, _ = prepare_digit(image)

    probabilities = model.predict(
        np.concatenate([original, prepared]), verbose=0
    )

    results.append({
        "Actual": digit,
        "Original prediction": int(probabilities[0].argmax()),
        "Prepared prediction": int(probabilities[1].argmax()),
    })

print(pd.DataFrame(results).to_string(index=False))