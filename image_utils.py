import numpy as np
from PIL import Image, ImageOps


def prepare_digit(image):
    gray = ImageOps.exif_transpose(image).convert("L")
    pixels = np.asarray(gray, dtype="float32")

    border = np.concatenate([
        pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1],
    ])

    # Conservar los píxeles de imágenes MNIST ya preparadas.
    if gray.size == (28, 28) and border.max() <= 30:
        return pixels.reshape(1, 784) / 255.0, gray

    # Convertir el fondo claro en oscuro.
    if np.median(border) > 127:
        pixels = 255.0 - pixels

    border = np.concatenate([
        pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1],
    ])

    # Quitar el fondo y reducir ruido sin binarizar el trazo.
    pixels = np.clip(pixels - np.median(border), 0, 255)
    peak = float(pixels.max())

    if peak < 20:
        raise ValueError("No digit detected.")

    pixels[pixels < max(10, peak * 0.08)] = 0
    mask = pixels > max(20, peak * 0.25)
    rows, columns = np.where(mask)

    if len(rows) == 0:
        raise ValueError("No digit detected.")

    # Recortar con margen para conservar los bordes suaves.
    top = max(0, int(rows.min()) - 2)
    bottom = min(pixels.shape[0], int(rows.max()) + 3)
    left = max(0, int(columns.min()) - 2)
    right = min(pixels.shape[1], int(columns.max()) + 3)

    cropped = pixels[top:bottom, left:right]
    cropped = np.clip(cropped / peak * 255, 0, 255)
    digit = Image.fromarray(cropped.astype("uint8"))

    # Ajustar tamaño conservando las proporciones.
    scale = 20 / max(digit.size)
    size = tuple(max(1, round(length * scale)) for length in digit.size)
    digit = digit.resize(size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(
        digit,
        ((28 - digit.width) // 2, (28 - digit.height) // 2),
    )

    # Centrar según la distribución de intensidad del dígito.
    values = np.asarray(canvas, dtype="float32")
    total = values.sum()
    yy, xx = np.indices(values.shape)

    center_x = float((values * xx).sum() / total)
    center_y = float((values * yy).sum() / total)
    shift_x = round(13.5 - center_x)
    shift_y = round(13.5 - center_y)

    # Limitar el desplazamiento para evitar cortar trazos.
    occupied_y, occupied_x = np.where(values > 0)
    shift_x = int(np.clip(
        shift_x, -occupied_x.min(), 27 - occupied_x.max()
    ))
    shift_y = int(np.clip(
        shift_y, -occupied_y.min(), 27 - occupied_y.max()
    ))

    centered = Image.new("L", (28, 28), 0)
    centered.paste(canvas, (shift_x, shift_y))

    features = np.asarray(
        centered, dtype="float32"
    ).reshape(1, 784) / 255.0

    return features, centered