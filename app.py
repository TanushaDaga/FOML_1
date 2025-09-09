import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("mnist_model.h5")

st.title("🖌️ Handwritten Digit Classifier")

uploaded_file = st.file_uploader("Upload an image of a digit (28x28 grayscale)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.write(f"### ✅ Predicted Digit: {predicted_class}")
