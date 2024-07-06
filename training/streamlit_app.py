import os
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

# Set the page configuration
st.set_page_config(
    page_title="Potato Disease Classification",
    page_icon=":potato:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Function to get the model path by navigating to the previous directory
def get_model_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_path = os.path.join('saved_models', '1.keras')
    return model_path

# Function to load the model
@st.cache_resource
def load_model():
    model_path = get_model_path()
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

st.write("""
         # Potato Disease Classification
         """)

# File uploader widget
file = st.file_uploader("Upload a potato leaf image", type=["jpg", "png", "jpeg"])

def import_and_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]  # 3D -> 4D image
    predictions = model.predict(img_reshape)
    class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

if file is None:
    st.text("Please upload a file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predicted_class, confidence = import_and_predict(image, model)
    string = f"Predicted label: {predicted_class}\n\rConfidence: {confidence}%"
    st.success(string)
