import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP



class DimensionalityReducer:
    def __init__(self, data, normalize=False, labels=None):
            self.data = data.copy()
            self.normalize = normalize
            self.labels = labels

            # Guardar los resultados cuando se aplique la reducción.
            self.reduced_data = None
            self.model = None

    def prepare_data(self):
            # Excluir la etiqueta y el índice; conservar variables numéricas.
            features = self.data.drop(
                columns=["Target", "Unnamed: 0"], errors="ignore"
            ).select_dtypes(include="number")

            if features.empty:
                raise ValueError("No numeric features found.")

            # Los píxeles siempre vacíos no aportan información.
            features = features.dropna(axis=1, how="all")

            if features.shape[1] < 2:
                raise ValueError("At least two numeric features are required.")

            # Completar valores faltantes con la mediana.
            self.imputer = SimpleImputer(strategy="median")
            self.prepared_data = self.imputer.fit_transform(features)

            # Escalar solo si se solicita.
            if self.normalize:
                self.scaler = StandardScaler()
                self.prepared_data = self.scaler.fit_transform(
                    self.prepared_data
                )

            return self.prepared_data
        
        # Reduce con LDA
    def lda(self):
            if self.labels is None:
                raise ValueError("LDA requires labels.")
            # Tuples to Numpy arrays 
            labels = np.array(self.labels)

            if labels.shape != (len(self.data),):
                raise ValueError("Provide one label per row.")
            
            # Indentifica missing values 
            if pd.isna(labels).any():
                raise ValueError("Labels must not contain missing values.")

            # Dos componentes requieren al menos tres clases.
            class_count = len(np.unique(labels))
            if class_count < 3:
                raise ValueError("LDA requires at least three classes.")

            if len(labels) <= class_count:
                raise ValueError("LDA requires more rows than classes.")

            data = self.prepare_data()
            self.model = LDA(n_components=2)
            result = self.model.fit_transform(data, labels)

            if result.shape[1] != 2:
                raise ValueError("The data cannot produce two LDA components.")

            self.reduced_data = result
            return self.reduced_data

            # Reduce con PCA
    def pca(self):
            data = self.prepare_data()

            if min(data.shape) < 2:
                raise ValueError("PCA requires at least two rows and features.")

            # Reducir los píxeles a dos componentes.
            self.model = PCA(n_components=2)
            self.reduced_data = self.model.fit_transform(data)

            return self.reduced_data
            
            # Reduce con t-SNE
    def tsne(self, perplexity=30, random_state=42):
            data = self.prepare_data()

            if not 0 < perplexity < len(data):
                raise ValueError(
                    "Perplexity must be positive and below the number of rows."
                )

            # Colocar cerca las imágenes con píxeles similares.
            self.model = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=random_state,
            )
            self.reduced_data = self.model.fit_transform(data)

            return self.reduced_data
        
            # Reduce con UMAP
    def umap(
        self, n_neighbors=15, min_dist=0.1, random_state=42
    ):
        data = self.prepare_data()

        if len(data) < 4:
            raise ValueError("UMAP requires at least four rows.")

        if not 2 <= n_neighbors < len(data):
            raise ValueError(
                "Neighbors must be at least two and below the row count."
            )

        if not 0 <= min_dist <= 1:
            raise ValueError("Minimum distance must be between zero and one.")

        # Reducir a dos componentes.
        self.model = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        self.reduced_data = self.model.fit_transform(data)
        return self.reduced_data

    def plot(self):
        if self.reduced_data is None:
            raise ValueError("Run a reduction method before plotting.")

        # Colorear por etiqueta cuando esté disponible.
        colors = None
        if self.labels is not None:
            colors, categories = pd.factorize(self.labels)

        fig, ax = plt.subplots(figsize=(8, 6))
        points = ax.scatter(
            self.reduced_data[:, 0],
            self.reduced_data[:, 1],
            c=colors,
            cmap="tab10" if colors is not None else None,
            s=10,
            alpha=0.6,
        )

        if colors is not None:
            colorbar = fig.colorbar(
                points, ax=ax, ticks=range(len(categories))
            )
            colorbar.ax.set_yticklabels(categories)
            colorbar.set_label("Digit")

        ax.set(
            title=type(self.model).__name__,
            xlabel="Component 1",
            ylabel="Component 2",
        )
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    # Mantener los nombres del enunciado y los usados por la interfaz.
    reduce_pca = pca
    reduce_lda = lda
    reduce_tsne = tsne
    reduce_umap = umap
