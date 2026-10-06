# Reducción de dimensionalidad y reconocimiento de dígitos

Proyecto práctico de ingeniería de características desarrollado en Python. Permite reducir datos tabulares a dos componentes mediante PCA, LDA, t-SNE y UMAP, y explorar los resultados mediante gráficos. Como extensión, incorpora una red neuronal sencilla para reconocer un dígito manuscrito a partir de una imagen.

La aplicación dispone de una interfaz de línea de comandos y un frontend en Streamlit. La reducción de dimensionalidad y el reconocimiento son funciones independientes: la red clasifica los 784 píxeles originales normalizados, no los dos componentes de una reducción.

App: https://dimreductionproyecto.streamlit.app/

## 1. Objetivos

- Construir la clase `DimensionalityReducer` para procesar datos tabulares.
- Ofrecer cuatro técnicas de reducción con salida de dos componentes.
- Permitir activar o desactivar el escalado de características.
- Visualizar resultados coloreados por etiqueta cuando esté disponible.
- Permitir elegir parámetros de t-SNE y UMAP desde el frontend.
- Entrenar una red neuronal con MNIST y guardar el modelo.
- Preparar imágenes propias y mostrar la predicción y las probabilidades por dígito.

## 2. Requisitos y dependencias

Se necesita Python, un entorno virtual y estas dependencias:

| Biblioteca | Uso |
|---|---|
| NumPy | Matrices, píxeles y operaciones numéricas |
| pandas | Lectura de CSV y manejo de tablas |
| scikit-learn | PCA, LDA, t-SNE, imputación, escalado y división de datos |
| umap-learn | Reducción mediante UMAP |
| Matplotlib | Gráficos de la CLI |
| Streamlit | Interfaz web local y gráficos interactivos |
| TensorFlow / Keras | Entrenamiento y carga de la red neuronal |
| Pillow | Lectura, transformación y exportación de imágenes |

Instalación desde el entorno virtual activo:

```powershell
python -m pip install numpy pandas scikit-learn umap-learn matplotlib streamlit tensorflow pillow
```

El entrenamiento registrado utilizó TensorFlow 2.21.0. No se ha fijado una versión de todas las dependencias; estos comandos no constituyen un entorno bloqueado para reproducibilidad exacta.

En el equipo utilizado, el entrenamiento se ejecutó con CPU. Una GPU no es necesaria para esta red pequeña. Con 8 GB de RAM, conviene cerrar servidores y aplicaciones que no se utilicen, y trabajar con la muestra reducida.

## 3. Estructura del proyecto

```text
dimreduction_proyecto/
├── app.py                  # Interfaz Streamlit
├── cli.py                  # Interfaz de línea de comandos
├── dim_reduction.py        # Clase y métodos de reducción
├── image_utils.py          # Preparación de imágenes
├── train_model.py          # Entrenamiento y guardado del modelo
├── mnist_model.keras       # Modelo entrenado
├── mnist_small.csv         # Dataset reducido
├── check_predictions.py    # Comparación antes/después del procesamiento
├── export_digits.py        # Exportación de ejemplos desde el CSV
├── test_reduction.py       # Prueba de UMAP con una muestra
├── README                  # Documentación del proyecto
├── mnist_examples/         # Dígitos PNG exportados para diagnóstico
├── numbers/                # Fotos propias
├── graficas/               # Gráficos guardados durante las pruebas
└── archive/                # Scripts históricos
```

`archive/` contiene `inspect_mnist.py` y `reduce_mnist.py`, que dependían del CSV grande eliminado. No son necesarios para ejecutar la aplicación actual. `__pycache__` es generado automáticamente por Python.

Los archivos necesarios para reconocimiento son `app.py`, `image_utils.py` y `mnist_model.keras`. La aplicación también importa `dim_reduction.py`, por lo que debe conservarse junto con sus dependencias. El CSV se carga solo en el modo de reducción; no es obligatorio para subir una foto.

## 4. Dataset

El CSV original de MNIST utilizado contenía 70 000 filas y 786 columnas:

- `pixel1` a `pixel784`: intensidades en escala de grises, entre 0 y 255.
- `Target`: etiqueta del dígito, entre 0 y 9.
- `Unnamed: 0`: índice exportado, sin utilidad como característica.

Cada imagen corresponde a una matriz de 28 × 28 píxeles, equivalente a 784 variables. No es necesario tomar fotografías con esa resolución: el procesamiento las adapta.

El archivo actual es `mnist_small.csv`, una muestra de 10 000 filas, con 1000 ejemplos por dígito. Al crearla se eliminó `Unnamed: 0`. Su tamaño observado fue aproximadamente 34 MB, lo que facilita las pruebas con memoria limitada. El CSV grande no es necesario para ejecutar los flujos actuales.

## 5. Preparación de datos tabulares

`DimensionalityReducer` recibe `data`, `normalize=False` y `labels=None`.

Su método `prepare_data()`:

1. Excluye `Target` y `Unnamed: 0` si existen.
2. Selecciona columnas numéricas.
3. Elimina columnas completamente vacías.
4. Comprueba que queden al menos dos variables.
5. Completa valores faltantes con la mediana de cada columna.
6. Aplica `StandardScaler` únicamente si `normalize=True`.
7. Guarda y devuelve `self.prepared_data`.

En el frontend, la columna seleccionada como etiqueta se separa de las variables antes de crear el objeto. Los métodos guardan su modelo en `self.model` y el resultado en `self.reduced_data`.

### Dos escalados diferentes

- Reducción de dimensiones: `StandardScaler` centra y escala cada variable; se activa opcionalmente.
- Clasificación de imágenes: dividir entre 255 convierte las intensidades a valores entre 0 y 1; es la escala usada al entrenar la red.

No se debe aplicar `StandardScaler` a una foto que se entregará al clasificador entrenado con división entre 255.

## 6. Métodos de reducción

| Método | Función | Etiquetas para ajustar | Condiciones principales |
|---|---|---|---|
| PCA | Proyección lineal que conserva variación | No | Al menos dos filas y dos variables |
| LDA | Proyección supervisada orientada a separar clases | Sí | Al menos tres clases para dos componentes y más filas que clases |
| t-SNE | Representación de relaciones locales | No | `0 < perplexity < número de filas` |
| UMAP | Representación basada en vecindad | No | En esta implementación, al menos cuatro filas y `2 <= n_neighbors < filas` |

Los cuatro métodos producen dos columnas. Las etiquetas pueden usarse para colorear los puntos incluso cuando no participan en el ajuste.

Los nombres utilizados por la aplicación son `pca()`, `lda()`, `tsne()` y `umap()`. También se exponen los alias solicitados en el enunciado: `reduce_with_pca()`, `reduce_with_lda()`, `reduce_with_tsne()` y `reduce_with_umap()`.

Parámetros disponibles:

- t-SNE: `perplexity=30` y `random_state=42`.
- UMAP: `n_neighbors=15`, `min_dist=0.1` y `random_state=42`.
- En esta implementación, `min_dist` se limita al intervalo de 0 a 1.
- PCA y LDA utilizan dos componentes fijos.

Los gráficos permiten explorar los datos, pero no representan por sí solos la precisión del clasificador. La escala y separación aparente de los gráficos de distintos métodos no deben interpretarse como medidas directamente equivalentes.

## 7. Ejecución desde PowerShell

Si la terminal comienza en la carpeta principal `Terminal 34 Bootcamp`, activar el entorno y entrar al proyecto:

```powershell
.\.venv\Scripts\Activate.ps1
cd "Frontend\dimreduction_proyecto"
```

Ejecutar estas líneas por separado. Si la terminal ya está dentro del proyecto y muestra `(.venv)`, no es necesario repetirlas.

### CLI

```powershell
python cli.py -m pca -f "mnist_small.csv" --normalize
python cli.py -m lda -f "mnist_small.csv" --normalize
python cli.py -m tsne -f "mnist_small.csv" --normalize
python cli.py -m umap -f "mnist_small.csv" --normalize
```

Sin escalado:

```powershell
python cli.py -m pca -f "mnist_small.csv"
```

Argumentos:

| Argumento | Uso |
|---|---|
| `-m`, `--method` | Obligatorio: `pca`, `lda`, `tsne` o `umap` |
| `-f`, `--filename` | Obligatorio: ruta del CSV |
| `--normalize` | Activa `StandardScaler`; si se omite, no se escala |

Aunque el parser contiene un valor predeterminado para PCA, `required=True` obliga a indicar `-m`. La CLI actual requiere una columna `Target` y utiliza los parámetros predeterminados de los métodos; los controles específicos están disponibles en Streamlit.

Cada ejecución imprime la forma del resultado y abre un gráfico de Matplotlib. Cerrar el gráfico permite finalizar la ejecución.

### Streamlit

```powershell
python -m streamlit run app.py --server.port=8503 --server.maxUploadSize=1000
```

Abrir `http://localhost:8503` y mantener la terminal activa. El límite de subida configurado es de 1000 MB; no equivale a garantizar memoria suficiente para cualquier archivo.

No ejecutar `app.py` con `python app.py` ni mediante un botón que use Python directamente: necesita el servidor de Streamlit. Para detenerlo, presionar Ctrl + C en la terminal donde está corriendo.

## 8. Uso del frontend

### Dimensionality reduction

1. Seleccionar la opción en la barra lateral.
2. Subir `mnist_small.csv`.
3. Revisar las primeras filas y las dimensiones de la tabla.
4. Elegir PCA, LDA, t-SNE o UMAP.
5. Activar o desactivar `Scale features`.
6. Elegir la cantidad de filas; el valor inicial es hasta 1000.
7. Seleccionar `Target` como etiqueta para MNIST; LDA necesita una etiqueta.
8. Configurar `Perplexity` o los parámetros de UMAP cuando corresponda.
9. Presionar `Run`.

Se toma una muestra con `random_state=42`, se ejecuta el método y se muestran los dos componentes en un gráfico interactivo. El CSV se conserva en caché para evitar leerlo en cada cambio de control.

### Digit recognition

1. Seleccionar la opción en la barra lateral.
2. Subir una imagen PNG, JPG o JPEG con un solo dígito.
3. Comparar la imagen original con la preparada de 28 × 28.
4. Presionar `Predict`.
5. Revisar el dígito predicho y el gráfico de probabilidades para las diez clases.

No es necesario cargar un CSV. Las fotos pueden estar en cualquier carpeta; `numbers/` es solo una opción de organización. El modelo se carga mediante caché y no se entrena al subir una imagen.

## 9. Preparación de fotos

La función `prepare_digit()` de `image_utils.py`:

1. Corrige la orientación EXIF y convierte a escala de grises.
2. Conserva imágenes de 28 × 28 cuyos bordes son oscuros, evitando alterar ejemplos MNIST ya preparados.
3. Estima el fondo a partir de los bordes e invierte las intensidades si es claro.
4. Resta el fondo estimado y elimina ruido débil.
5. Localiza el dígito con una máscara y recorta con margen, sin binarizar completamente el trazo.
6. Ajusta el contraste y conserva la relación de aspecto al reducir el lado mayor a unos 20 píxeles.
7. Coloca el resultado en un lienzo negro de 28 × 28.
8. Centra según la distribución de intensidad, limitando el desplazamiento para no cortar trazos.
9. Convierte a una entrada de forma `(1, 784)` y divide entre 255.

La detección del fondo supone que los bordes representan el fondo de la imagen. Para obtener mejores entradas, usar un solo dígito, iluminación uniforme, trazos completos y margen alrededor. Si hay una mesa oscura alrededor de papel claro, recortar dentro del papel antes de subirlo.

## 10. Red neuronal

El clasificador está implementado con TensorFlow y Keras:

```text
784 intensidades → Dense(128, ReLU) → Dense(10, Softmax)
```

Tiene 101 770 parámetros entrenables. La salida contiene diez probabilidades; se elige la clase con el valor más alto. La red es densa, no una CNN, y no utiliza aumento de datos en la versión actual.

Configuración de entrenamiento:

| Parámetro | Valor |
|---|---|
| Optimizador | Adam |
| Pérdida | sparse_categorical_crossentropy |
| Métrica | accuracy |
| Épocas máximas | 15 |
| Tamaño de lote | 128 |
| Semilla | 42 |
| Detención temprana | val_loss, patience=2 |
| Recuperación de pesos | Mejores pesos de validación |

Para 10 000 filas, la división estratificada es:

| Conjunto | Filas | Porcentaje |
|---|---|---|
| Entrenamiento | 7000 | 70 % |
| Validación | 1000 | 10 % |
| Prueba | 2000 | 20 % |

Se separa primero el 20 % para prueba y después el 12,5 % de las filas restantes para validación. El conjunto de prueba no se entrega a `model.fit()`.

Entrenar:

```powershell
python train_model.py
```

El script actual lee `mnist_small.csv`, selecciona hasta 10 000 filas y guarda `mnist_model.keras` junto al script. Ejecutarlo reemplaza el modelo guardado: no es necesario hacerlo para abrir el frontend. La puntuación histórica indicada abajo no garantiza el mismo resultado al reentrenar con una muestra diferente.

## 11. Resultados y validación

### Reducción

Durante el desarrollo se ejecutaron PCA, LDA, t-SNE y UMAP y se guardaron sus gráficos. PCA produjo `(70000, 2)` cuando se utilizó el CSV completo. Ahora el número de filas depende de `mnist_small.csv` y de la muestra seleccionada; con 1000 filas, la salida esperada es `(1000, 2)`.

### Clasificador

En el entrenamiento registrado con una muestra aleatoria de 10 000 imágenes del CSV original:

- Exactitud final de entrenamiento: 98,57 %.
- Exactitud final de validación: 93,50 %.
- Exactitud en prueba: 94,15 % sobre 2000 imágenes.
- Modelo guardado: `mnist_model.keras`.

La muestra reducida actual se construyó posteriormente. Reentrenar con ella constituye una nueva ejecución y requiere volver a registrar sus métricas.

### Diagnóstico de imágenes

Antes del ajuste, algunas imágenes exportadas del CSV cambiaban de predicción después de recortarlas y centrarlas. Se corrigió el flujo para conservar imágenes MNIST de 28 × 28 con fondo negro. En la comparación registrada, los diez ejemplos exportados produjeron la clase correcta tanto directamente como después de `prepare_digit()`.

Esos diez ejemplos pueden haber participado en el entrenamiento. Son una prueba del flujo de procesamiento, no una evaluación independiente de generalización.

Tras mejorar la limpieza y el centrado por intensidad, la usuaria confirmó reconocimiento correcto con sus fotos y cumplimiento de la prueba mínima de dos imágenes propias. No se registró una tasa global de acierto ni probabilidades finales para todas esas fotos; por tanto, no se atribuye un porcentaje de precisión a fotos nuevas.

La probabilidad de Softmax mostrada para una imagen no es la exactitud del modelo y no garantiza que la predicción sea correcta.

## 12. Scripts de comprobación

Exportar un ejemplo de cada dígito desde el CSV reducido:

```powershell
python export_digits.py
```

Se leen bloques de 500 filas y se guardan `digit_0.png` a `digit_9.png` en `mnist_examples/`.

Comparar predicciones directas y procesadas de los mismos ejemplos:

```powershell
python check_predictions.py
```

La tabla contiene `Actual`, `Original prediction` y `Prepared prediction`. Permite detectar cambios introducidos por el procesamiento.

Probar reducción con hasta 1000 filas:

```powershell
python test_reduction.py
```

En su versión actual utiliza UMAP y abre el gráfico con Matplotlib. Estos scripts son auxiliares: no son necesarios para abrir Streamlit.

## 13. Solución de problemas

| Mensaje o situación | Acción |
|---|---|
| `File does not exist: app.py` | Entrar a la carpeta del proyecto antes de iniciar Streamlit |
| `can't open file` | Revisar la carpeta actual y el nombre del script |
| CSV no encontrado | Usar `mnist_small.csv` y comprobar su ubicación |
| `No module named Frontend` | Usar imports locales, como `from image_utils import prepare_digit` |
| `ModuleNotFoundError` de una biblioteca | Activar el entorno e instalar la dependencia con `python -m pip` |
| Faltan `-m` y `-f` | Ejecutar la CLI con ambos argumentos |
| `missing ScriptRunContext` | Ejecutar con `python -m streamlit run app.py` |
| Puerto ocupado | Detener el servidor anterior en su terminal o utilizar otro puerto |
| Aviso de oneDNN | Es informativo; no significa que el entrenamiento haya fallado |
| Aviso de GPU en Windows | Esta ejecución utiliza CPU; no es necesario configurar GPU para la prueba |
| Aviso de UMAP sobre random_state | La semilla limita el paralelismo; la reducción puede continuar |
| `No digit detected` | Revisar contraste, iluminación y contenido de la imagen |
| Foto predicha incorrectamente | Revisar primero la vista preparada; probar ejemplos de control y registrar resultados |

Con memoria limitada, mantener un solo servidor Streamlit, detenerlo antes de entrenar y comenzar con 1000 filas para reducción. La aplicación mantiene datos y modelo en caché mientras el proceso está activo; borrar archivos temporales del disco no sustituye liberar RAM.

## 14. Alcance y limitaciones

- Reconoce un solo dígito manuscrito por imagen, no números de varios dígitos.
- No garantiza aciertos para cualquier estilo de escritura, fondo, sombra o inclinación.
- La preparación automática puede fallar si los bordes no representan el fondo.
- La prueba mínima de fotos propias confirma funcionamiento puntual, no una precisión universal.
- El proyecto se ejecuta localmente; no incluye publicación web ni autenticación.
- La CLI está orientada al esquema MNIST con `Target`; el frontend permite seleccionar otras etiquetas.
- La preparación excluye columnas de texto; no incorpora codificación de variables categóricas.
- La implementación valida condiciones básicas, pero no todos los posibles casos degenerados de los datos.

## 15. Evidencias para la entrega

- Código de la clase, CLI y frontend.
- Gráficos de los cuatro métodos.
- Registro del entrenamiento y exactitud de prueba.
- Capturas de al menos dos fotos propias reconocidas, mostrando original, preparación, predicción y probabilidad.
- Este README con comandos, dependencias, resultados y limitaciones.

## Autor y versión

- Autora: Lariza Amorós.
- Correo: lariza.amoros@gmail.com.
- Versión del proyecto: 2.1.
- Documentación actualizada: 5 de octubre de 2026.
