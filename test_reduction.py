from pathlib import Path
import pandas as pd
from dim_reduction import DimensionalityReducer

# Usar una muestra para comprobar el funcionamiento.
data = pd.read_csv(Path(__file__).parent / "mnist_small.csv", nrows=1000)
reducer = DimensionalityReducer(
    data=data,
    normalize=True,
    labels=data["Target"],
    )

result = reducer.umap()
print(f"Reduced data shape: {result.shape}")
reducer.plot()
