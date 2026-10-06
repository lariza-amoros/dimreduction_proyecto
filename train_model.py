from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


# Localizar el CSV junto a este archivo.
csv_path = Path(__file__).parent / "mnist_small.csv"
data = pd.read_csv(csv_path, dtype={f"pixel{i}": "float32" for i in range(1, 785)})

# Usar una muestra para reducir el consumo de memoria.
data = data.sample(n=min(10000, len(data)), random_state=42).copy()

# Separar los píxeles y normalizarlos entre 0 y 1.
X = data.filter(regex=r"^pixel\d+$").to_numpy(dtype="float32") / 255.0
y = data["Target"].to_numpy(dtype="int64")

# Reservar el 20 % para la evaluación final.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

# Separar validación sin utilizar el conjunto de prueba.
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.125,
    stratify=y_train,
    random_state=42,
)

print(f"Training: {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test: {X_test.shape}")

# Fijar la semilla para facilitar la reproducibilidad.
tf.keras.utils.set_random_seed(42)

# Recibir 784 píxeles y producir probabilidades para los 10 dígitos.
model = tf.keras.Sequential([
    tf.keras.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

# Configurar cómo aprende la red.
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Detener cuando la validación deje de mejorar.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=128,
    callbacks=[early_stopping],
)

# Evaluar una vez con datos que no participaron en el entrenamiento.
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.2%}")

# Guardar el modelo junto al script.
model_path = Path(__file__).parent / "mnist_model.keras"
model.save(model_path)
print(f"Model saved: {model_path.name}")
