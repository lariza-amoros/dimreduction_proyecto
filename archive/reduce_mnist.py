from pathlib import Path

import pandas as pd


base = Path(__file__).parent
source = base / "mnist_tabular.csv"
destination = base / "mnist_small.csv"

counts = {digit: 0 for digit in range(10)}
write_header = True

# Leer bloques pequeños y guardar una muestra equilibrada.
for chunk in pd.read_csv(source, chunksize=500):
    parts = []

    for digit in range(10):
        remaining = 1000 - counts[digit]
        if remaining == 0:
            continue

        selected = chunk[chunk["Target"] == digit].head(remaining)
        counts[digit] += len(selected)
        parts.append(selected)

    if parts:
        sample = pd.concat(parts).drop(
            columns=["Unnamed: 0"], errors="ignore"
        )
        sample.to_csv(
            destination,
            mode="w" if write_header else "a",
            header=write_header,
            index=False,
        )
        write_header = False

    if all(count == 1000 for count in counts.values()):
        break

print(f"Saved rows: {sum(counts.values())}")
print(f"File: {destination.name}")