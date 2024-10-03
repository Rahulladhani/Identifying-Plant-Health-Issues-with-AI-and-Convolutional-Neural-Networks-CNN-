import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import time
import gdown  # Added to download files from Google Drive

# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress detailed TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Google Drive URL for the model file (your actual file ID from the link)git add requirements.txt
# git commit -m "Added TensorFlow to requirements"
# git push origin main
model_url = 'https://drive.google.com/uc?id=1hJTry62PBf9fCV6namdVa9qzTvf8sEBl'
output_model = 'plant_disease_prediction_model.h5'

# Download the model file if it doesn't exist locally
if not os.path.exists(output_model):
    gdown.download(model_url, output_model, quiet=False)

# Path to the class indices file
class_indices_path = os.path.join(os.path.dirname(__file__), 'class_indices.json')

# Function to load the model with error handling
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to load class indices with error handling
def load_class_indices(class_indices_path):
    try:
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        print("Class indices loaded successfully.")
        return class_indices
    except FileNotFoundError:
        print(f"Error: {class_indices_path} not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Failed to parse class indices JSON.")
        return None

# Load the pre-trained model and class indices
model = load_model(output_model)
class_indices = load_class_indices(class_indices_path)

# Ensure both model and class indices are loaded successfully
if model is None or class_indices is None:
    st.error("Failed to load model or class indices. Please check your setup.")
    st.stop()  # Stops the app if there’s an error loading the model or class indices

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Sidebar for additional options
st.sidebar.title("Navigation")
st.sidebar.write("Use the buttons below to explore the app.")
st.sidebar.button("Upload Image")
st.sidebar.button("Predict")
st.sidebar.button("About Us")

# Page Title and Instructions
st.markdown("<h1 style='text-align: center; color: green;'>Plant Disease Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>Upload an image to detect plant diseases using AI</h4>", unsafe_allow_html=True)

# Instructions with tooltip
st.markdown(
    "<h4>Instructions <span style='color: grey;' title='Step-by-step guide on how to use the app.'> ⓘ</span></h4>",
    unsafe_allow_html=True)
st.markdown("""
1. Upload a clear image of the plant leaf.
2. Wait for the model to process and classify the disease.
3. The results will appear below along with some advice.
""")

# Image Upload
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Uploaded Image:")
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Show a progress bar while processing the image
            with st.spinner('Classifying the image...'):
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(percent_complete + 1)

            # Ensure the model is not None before prediction
            if model is not None and class_indices is not None:
                prediction = predict_image_class(model, image, class_indices)
                st.success(f'Prediction: {prediction}')
                st.markdown("<h3 style='color:blue;'>Advice: Make sure to consult an expert for detailed guidance!</h3>", unsafe_allow_html=True)
            else:
                st.error("Model or class indices are not loaded. Cannot classify the image.")

# Footer with a custom message
st.markdown("---")
st.markdown("<h4 style='text-align: center; color: grey;'>Powered by AI - Plant Disease Detection</h4>", unsafe_allow_html=True)
