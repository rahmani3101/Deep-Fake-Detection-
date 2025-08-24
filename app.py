import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

# Set numpy to suppress scientific notation
np.set_printoptions(suppress=True)

# Load the model
@st.cache_resource
def load_keras_model():
    return load_model('asad.h5', compile=False)

# Define class names (replace with your actual class names)
class_names = ['Real', 'Fake']  

# Streamlit app
st.title("Image Classification with Keras Model")
st.write("Upload an image to classify it using the pre-trained Keras model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and process the image
    image = Image.open(uploaded_file).convert('RGB')
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Convert image to numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Prepare data for prediction
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Load model and predict
    model = load_keras_model()
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Display results
    st.write(f"**Class:** {class_name}")
    st.write(f"**Confidence Score:** {confidence_score:.4f}")

    # Plot the image with predicted class as title
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(f"Predicted: {class_name}")
    ax.axis('off')
    st.pyplot(fig)

else:
    st.write('No File Uploaded !!')     