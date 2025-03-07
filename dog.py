import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('dog_breed_classifier.h5')

model = load_model()

# Define class names
class_names = ['Affenhuahua dog', 'Afgan Hound dog', 'Akita dog', 'Alaskan Malamute dog',
               'American Bulldog dog', 'Auggie dog', 'Beagle dog', 'Belgian Tervuren dog',
               'Bichon Frise dog', 'Bocker dog', 'Borzoi dog', 'Boxer dog',
               'Bugg dog', 'Bulldog dog']

# Title
st.title("🐶 Dog Breed Classifier")
st.write("Upload a dog image, and the model will predict its breed!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def prepare_image(image):
    image = image.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner('Classifying...'):
        prepared_image = prepare_image(image)
        prediction = model.predict(prepared_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    st.success(f"### 🐕 Predicted Breed: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
