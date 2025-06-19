import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import io

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('emnist_model.h5')
    return model

model = load_model()

# Define the mapping of class indices to characters for EMNIST ByClass
# This mapping is based on the dataset documentation
emnist_classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

st.title("EMNIST Character Recognition")
st.write("Upload an image of a handwritten character (0-9, A-Z, a-z) for recognition.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the image file
    image = Image.open(uploaded_file).convert('L') # Convert to grayscale

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for the model
    # Resize to 28x28
    image = image.resize((28, 28))
    # Invert colors if necessary (EMNIST has white characters on black background)
    image = ImageOps.invert(image)
    # Convert to numpy array and normalize
    img_array = np.array(image).astype('float32') / 255.0
    # Reshape for the model (add batch dimension and channel dimension)
    img_array = img_array.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_character = emnist_classes[predicted_class_index]

    st.write(f"Predicted Character: **{predicted_character}**")
