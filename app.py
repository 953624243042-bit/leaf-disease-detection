import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🌿 Leaf Disease Detection")
st.write("Upload a plant leaf image to detect the disease.")

# --------------------------------------------------
# Model Path
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "leaf_disease_cnn_model.h5"
)

# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file not found. "
            "Please upload leaf_disease_cnn_model.h5 "
            "to the GitHub repository."
        )
        st.stop()

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


model = load_model()

# --------------------------------------------------
# Class Names
# --------------------------------------------------

classes = [
    "healthy",
    "multiple_diseases",
    "rust",
    "scab"
]

# --------------------------------------------------
# Upload Image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    # --------------------------------------------------
    # Preprocessing
    # Same as Colab
    # --------------------------------------------------

    img = image.resize((224, 224))

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    with st.spinner("Predicting disease..."):

        prediction = model.predict(
            img_array,
            verbose=0
        )

    # --------------------------------------------------
    # Get Prediction
    # --------------------------------------------------

    predicted_index = np.argmax(
        prediction[0]
    )

    predicted_class = classes[
        predicted_index
    ]

    confidence = (
        prediction[0][predicted_index] * 100
    )

    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.subheader("Prediction Result")

    st.success(
        f"Predicted Disease: {predicted_class}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    # --------------------------------------------------
    # All Class Probabilities
    # --------------------------------------------------

    st.subheader("Prediction Probabilities")

    for i, class_name in enumerate(classes):

        probability = (
            prediction[0][i] * 100
        )

        st.write(
            f"{class_name}: {probability:.2f}%"
        )

        st.progress(
            float(prediction[0][i])
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Leaf Disease Detection using CNN"
)
