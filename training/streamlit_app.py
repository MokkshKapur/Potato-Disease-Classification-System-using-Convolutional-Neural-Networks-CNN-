import streamlit as st
import tensorflow as tf

import cv2
from PIL import Image, ImageOps
import numpy as np

st.setoption('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('D:\_programming\Potato-Disease-Classification-System-using-Convolutional-Neural-Networks-CNN\saved_models\1.keras')
    return model

model = load_model()
st.write("""
         # Potato Disease Classification
         """)
file = st.file_uploader("Upload a potato leaf image", type=["jpg","png","jpeg"])
def import_and_predict(image_data, model):
    size = (256,256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]   #3D -> 4D image
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
    string = f"predicted label : {predicted_class}\nconfidence : {confidence}%"
    st.success(string)
