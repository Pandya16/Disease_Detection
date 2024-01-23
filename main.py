import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle

file_path = "diseases.pkl"
with open(file_path, "rb") as f:
    model = pickle.load(f)
class_labels = ['chickenpox', 'measles', 'monkeypox', 'normal']  

st.title('Disease Detection')

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))  
    image_array = np.asarray(image)
    image_array = image_array / 255.0  
    image_array = np.expand_dims(image_array, axis=0)  

    predictions = model.predict(image_array)
    
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Prediction:")
    st.write(class_labels[np.argmax(predictions)])


