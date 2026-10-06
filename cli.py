# Propósito: Redución de dimensión de datos utilizando PCA, LDA, t-SNE y UMAP
# Autor: Lariza Amorós, lariza.amoros@gmail.com
# Versión: 2.2 10/02/2026

import argparse
import pandas as pd

from dim_reduction import DimensionalityReducer


def main():
    parser = argparse.ArgumentParser(
        description="Dimensionality reduction app"
    )

    parser.add_argument(
        "-m", "--method",
        choices=["pca", "lda", "tsne", "umap"],
        default="pca",
    )
    parser.add_argument(
        "-f", "--filename", required=True,
        help="Input CSV file path",
    )
    parser.add_argument(
        "--target",
        help="Label column; required for LDA",
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Scale features before reduction",
    )
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)

    args = parser.parse_args()

    # LDA necesita etiquetas; los demás métodos no.
    if args.method == "lda" and args.target is None:
        parser.error("LDA requires --target.")

    try:
        data = pd.read_csv(args.filename)
        labels = None

        # Separar la etiqueta para evitar usarla como característica.
        if args.target is not None:
            if args.target not in data.columns:
                raise ValueError(f"Column not found: {args.target}")
            labels = data.pop(args.target)

        # Target es la etiqueta reservada de MNIST.
        if "Target" in data.columns:
            data = data.drop(columns=["Target"])

        reducer = DimensionalityReducer(
            data=data,
            normalize=args.normalize,
            labels=labels,
        )

        if args.method == "pca":
            result = reducer.pca()
        elif args.method == "lda":
            result = reducer.lda()
        elif args.method == "tsne":
            result = reducer.tsne(
                perplexity=args.perplexity
            )
        else:
            result = reducer.umap(
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist,
            )

        print(f"Method: {args.method}")
        print(f"Normalization: {args.normalize}")
        print(f"Reduced data shape: {result.shape}")
        reducer.plot()

    except (ValueError, OSError, pd.errors.ParserError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    print("Dimensionality reduction application")
    main()