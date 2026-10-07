import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🌿 Leaf Disease Detection")
st.write("Upload a leaf image to detect the disease using CNN.")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("leaf_disease_cnn_model.h5")
    return model

model = load_model()

# -----------------------------
# Class Names
# -----------------------------
class_names = [
    "Healthy",
    "Multiple Diseases",
    "Rust",
    "Scab"
]

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    # -----------------------------
    # Preprocessing
    # -----------------------------
    img = image.resize((224, 224))

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction[0])

    confidence = np.max(prediction[0]) * 100

    # -----------------------------
    # Display Result
    # -----------------------------
    st.subheader("Prediction Result")

    st.success(
        f"Predicted Disease: {class_names[predicted_class]}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    # -----------------------------
    # Prediction Probabilities
    # -----------------------------
    st.subheader("Prediction Probabilities")

    for i, class_name in enumerate(class_names):
        probability = prediction[0][i] * 100

        st.write(
            f"{class_name}: {probability:.2f}%"
        )

        st.progress(float(prediction[0][i]))
