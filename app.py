import pandas as pd
import streamlit as st
from dim_reduction import DimensionalityReducer
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from image_utils import prepare_digit

st.set_page_config(page_title="Dimensionality Reducer", layout="wide")
st.title("Dimensionality Reducer")

# Elegir la función de la aplicación antes de solicitar archivos.
task = st.sidebar.radio(
    "Choose a task",
    ["Dimensionality reduction", "Digit recognition"],
)

if task == "Digit recognition":
    st.header("Digit recognition")
    st.caption("Upload an image containing one handwritten digit.")

    # Cargar el modelo una sola vez y solo para reconocimiento.
    @st.cache_resource
    def load_digit_model():
        import tensorflow as tf

        model_path = Path(__file__).parent / "mnist_model.keras"
        return tf.keras.models.load_model(model_path, compile=False)

    image_file = st.file_uploader(
        "Upload a digit image",
        type=["png", "jpg", "jpeg"],
        key="digit_image",
    )

    if image_file is not None:
        try:
            with Image.open(image_file) as image:
                features, preview = prepare_digit(image)
                original = image.convert("RGB")

            # Mostrar lo que ve el usuario y lo que recibe el modelo.
            left, right = st.columns(2)
            left.image(original, caption="Original image", width=250)
            right.image(preview, caption="Prepared image: 28 × 28", width=250)

            if st.button("Predict"):
                with st.spinner("Recognizing digit..."):
                    model = load_digit_model()
                    probabilities = model.predict(features, verbose=0)[0]

                prediction = int(np.argmax(probabilities))
                st.metric("Predicted digit", prediction)
                st.write(
                    f"Model probability: {probabilities[prediction]:.1%}"
                )

                scores = pd.DataFrame({
                    "Digit": [str(digit) for digit in range(10)],
                    "Probability": probabilities,
                })
                st.bar_chart(scores, x="Digit", y="Probability")

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            ImportError,
        ) as error:
            st.error(f"Unable to predict: {error}")

    # Evitar solicitar un CSV en el modo de reconocimiento.
    st.stop()

upload_file = st.file_uploader("Upload a CSV file", type=["csv"])

if upload_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

# Evitar leer el CSV nuevamente al cambiar los controles.
@st.cache_data
def load_data(file):
    return pd.read_csv(file)


try:
    data = load_data(upload_file)
except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
    st.error(f"Unable to read CSV: {error}")
    st.stop()

st.dataframe(data.head())
st.write(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")

if len(data) < 4:
    st.error("Upload a dataset with at least four rows.")
    st.stop()

# Configurar el método y la muestra.
method = st.selectbox("Reduction method", ["PCA", "LDA", "t-SNE", "UMAP"])
normalize = st.checkbox("Scale features", value=True)

sample_size = st.number_input(
    "Number of rows to process",
    min_value=4,
    max_value=len(data),
    value=min(1000, len(data)),
    step=1,
)

# Usar None como opción sin etiqueta.
label_options = [None] + data.columns.tolist()
label_column = st.selectbox(
    "Label column",
    label_options,
    index=label_options.index("Target") if "Target" in data.columns else 0,
    format_func=lambda column: "None" if column is None else str(column),
)

# Mostrar los parámetros correspondientes al método.
if method == "t-SNE":
    perplexity = st.number_input(
        "Perplexity",
        min_value=1.0,
        max_value=float(sample_size - 1),
        value=min(30.0, float(sample_size - 1)),
    )

elif method == "UMAP":
    n_neighbors = st.number_input(
        "Number of neighbors",
        min_value=2,
        max_value=int(sample_size - 1),
        value=min(15, int(sample_size - 1)),
    )
    min_dist = st.slider("Minimum distance", 0.0, 1.0, 0.1)

if st.button("Run"):
    if method == "LDA" and label_column is None:
        st.error("Select a label column for LDA.")
        st.stop()

    # Mantener alineadas las variables y sus etiquetas.
    sample = data.sample(n=int(sample_size), random_state=42)
    features = sample.copy()
    labels = None

    if label_column is not None:
        labels = features.pop(label_column)

    reducer = DimensionalityReducer(
        data=features,
        normalize=normalize,
        labels=labels,
    )

    try:
        with st.spinner("Reducing dimensions..."):
            if method == "PCA":
                result = reducer.pca()
            elif method == "LDA":
                result = reducer.lda()
            elif method == "t-SNE":
                result = reducer.tsne(
                    perplexity=perplexity
                )
            else:
                result = reducer.umap(
                    n_neighbors=int(n_neighbors),
                    min_dist=min_dist,
                )

        # Mostrar los componentes dentro de Streamlit.
        chart = pd.DataFrame(
            result, columns=["Component 1", "Component 2"]
        )

        if labels is not None:
            chart["Label"] = labels.astype(str).to_numpy()

        st.scatter_chart(
            chart,
            x="Component 1",
            y="Component 2",
            color="Label" if labels is not None else None,
        )
        st.success(f"Reduced data shape: {result.shape}")

    except ValueError as error:
        st.error(str(error))