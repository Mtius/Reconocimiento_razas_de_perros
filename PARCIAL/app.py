import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = 224
DEFAULT_MODEL_PATH = Path("modelo_perros.keras")
DEFAULT_CLASS_NAMES_PATH = Path("class_names.json")


@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_class_names(class_names_path: str):
    with open(class_names_path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or len(names) == 0:
        raise ValueError("class_names.json debe contener una lista no vacia de razas.")
    return names


def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0), image


def top_k_predictions(predictions: np.ndarray, class_names: list[str], k: int = 5):
    idx = np.argsort(predictions)[::-1][:k]
    return [(class_names[i], float(predictions[i])) for i in idx]


st.set_page_config(page_title="Clasificador de Razas de Perros", page_icon="🐶", layout="centered")

st.title("Clasificador de Razas de Perros")
st.write(
    "Sube una imagen de un perro para predecir su raza usando tu modelo entrenado en Keras."
)

st.sidebar.header("Configuracion de archivos")
model_source = st.sidebar.radio(
    "Modelo",
    options=["Usar archivo local", "Subir archivo .keras"],
    index=0,
)
classes_source = st.sidebar.radio(
    "Nombres de razas",
    options=["Usar archivo local", "Subir class_names.json"],
    index=0,
)

model_path_to_load = None
class_names_path_to_load = None

if model_source == "Usar archivo local":
    if DEFAULT_MODEL_PATH.exists():
        model_path_to_load = str(DEFAULT_MODEL_PATH)
        st.sidebar.success(f"Modelo encontrado: {DEFAULT_MODEL_PATH}")
    else:
        st.sidebar.error("No se encontro modelo_perros.keras en la carpeta del proyecto.")
else:
    uploaded_model = st.sidebar.file_uploader("Sube tu modelo .keras", type=["keras"])
    if uploaded_model is not None:
        temp_model_path = Path("_uploaded_model.keras")
        temp_model_path.write_bytes(uploaded_model.getbuffer())
        model_path_to_load = str(temp_model_path)
        st.sidebar.success("Modelo cargado temporalmente.")

if classes_source == "Usar archivo local":
    if DEFAULT_CLASS_NAMES_PATH.exists():
        class_names_path_to_load = str(DEFAULT_CLASS_NAMES_PATH)
        st.sidebar.success(f"Clases encontradas: {DEFAULT_CLASS_NAMES_PATH}")
    else:
        st.sidebar.error("No se encontro class_names.json en la carpeta del proyecto.")
else:
    uploaded_classes = st.sidebar.file_uploader("Sube class_names.json", type=["json"])
    if uploaded_classes is not None:
        temp_classes_path = Path("_uploaded_class_names.json")
        temp_classes_path.write_bytes(uploaded_classes.getbuffer())
        class_names_path_to_load = str(temp_classes_path)
        st.sidebar.success("class_names.json cargado temporalmente.")

if model_path_to_load and class_names_path_to_load:
    try:
        model = load_model(model_path_to_load)
        class_names = load_class_names(class_names_path_to_load)
    except Exception as e:
        st.error(f"Error cargando archivos: {e}")
        st.stop()

    uploaded_image = st.file_uploader(
        "Sube una imagen de perro", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_image is not None:
        img_bytes = uploaded_image.read()
        image = Image.open(io.BytesIO(img_bytes))

        st.image(image, caption="Imagen subida", use_container_width=True)

        with st.spinner("Analizando imagen..."):
            image_batch, processed_image = preprocess_image(image)
            prediction = model.predict(image_batch, verbose=0)[0]
            top5 = top_k_predictions(prediction, class_names, k=5)

        best_breed, best_prob = top5[0]
        st.subheader(f"Prediccion principal: {best_breed}")
        st.write(f"Confianza: {best_prob * 100:.2f}%")

        st.markdown("### Top 5 predicciones")
        chart_data = {
            "Raza": [name for name, _ in top5],
            "Probabilidad": [prob for _, prob in top5],
        }
        st.bar_chart(chart_data, x="Raza", y="Probabilidad")

        with st.expander("Ver detalles"):
            for breed, prob in top5:
                st.write(f"- {breed}: {prob * 100:.2f}%")

        # Evita variables sin uso por claridad del flujo de inferencia.
        _ = processed_image
else:
    st.info(
        "Carga modelo_perros.keras y class_names.json para comenzar. "
        "Tambien puedes colocarlos en la raiz del proyecto para usar carga local."
    )
