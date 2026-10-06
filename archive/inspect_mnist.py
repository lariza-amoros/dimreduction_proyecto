import pandas as pd

# Leer una muestra y seleccionar únicamente los píxeles.
data = pd.read_csv("mnist_tabular.csv", nrows=1000)
pixels = data.filter(regex=r"^pixel\d+$")

print(f"Pixel columns: {pixels.shape[1]}")
print(f"Minimum: {pixels.min().min()}")
print(f"Maximum: {pixels.max().max()}")
print(f"Labels: {sorted(data['Target'].unique())}")