import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

st.set_page_config(
    page_title="Plant-Disease-Classifier | Agro-Pro | Team Indra",
    page_icon="🌿",
    layout="centered"
)

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/plant_disease_prediction_model_quantized.tflite"

# Initialize the TensorFlow Lite interpreter
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details from the interpreter
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image):
    target_size = (224, 224)
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.  # Normalize the image
    return img_array


# Function to Predict the Class of an Image using the TFLite Interpreter
def predict_image_class(interpreter, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)

    # Set the tensor with preprocessed image
    interpreter.set_tensor(input_details[0]['index'], preprocessed_img)

    # Run inference
    interpreter.invoke()

    # Get prediction results
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Map the predicted class index to the actual class name
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.title('🌿 Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(interpreter, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
