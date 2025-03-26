import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar modelo
model = tf.keras.models.load_model("modelo_final1.h5")

# Función para preprocesar la imagen
def preprocess_image(image):
    image = image.resize((224,224))  # Redimensionar a 128x128
    image = np.array(image) / 255.0   # Normalizar valores entre 0 y 1
    image = np.expand_dims(image, axis=0)  # Agregar dimensión batch (1, 128, 128, 3)
    return image

# Interfaz en Streamlit
st.title("Clasificación de Enfermedades en Cítricos 🍊")

uploaded_file = st.file_uploader("Sube una imagen de cítricos", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Preprocesar la imagen
    processed_image = preprocess_image(image)

    # Hacer la predicción
    prediction = model.predict(processed_image)

    # Obtener la clase con mayor probabilidad
    class_names = ["Black Spot", "Citrus Canker"]
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"**Enfermedad detectada:** {predicted_class} ✅")
