import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('dog_breed_classifier.h5')

# Define the class names (same order as during training)
class_names = ['Affenhuahua dog', 'Afgan Hound dog', 'Akita dog', 'Alaskan Malamute dog',
               'American Bulldog dog', 'Auggie dog', 'Beagle dog', 'Belgian Tervuren dog',
               'Bichon Frise dog', 'Bocker dog', 'Borzoi dog', 'Boxer dog', 'Bugg dog', 'Bulldog dog']

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Streamlit dashboard setup
st.title("🐶 Dog Breed Classifier")
st.write("Upload a dog image and the model will predict the breed!")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Show result
    st.success(f"Predicted Dog Breed: **{predicted_class}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
