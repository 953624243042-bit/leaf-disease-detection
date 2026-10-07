import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)


# =========================================================
# TITLE
# =========================================================

st.title("🌿 Leaf Disease Detection")

st.write(
    "Plant Pathology 2020 dataset analysis using Deep Learning."
)

st.info(
    "Upload the sample submission CSV to analyze the "
    "leaf disease prediction probabilities."
)


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = "leaf_disease_autoencoder.keras"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


model = load_model()


if model is None:

    st.error(
        "Model file not found. "
        "Please upload leaf_disease_autoencoder.keras "
        "to the GitHub repository."
    )

    st.stop()


# =========================================================
# UPLOAD CSV
# =========================================================

uploaded_file = st.file_uploader(
    "Upload sample_submission CSV",
    type=["csv"]
)


# =========================================================
# PROCESS CSV
# =========================================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        required_columns = [
            "image_id",
            "healthy",
            "multiple_diseases",
            "rust",
            "scab"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:

            st.error(
                "Required columns are missing: "
                + ", ".join(missing_columns)
            )

            st.stop()


        # =================================================
        # DISEASE COLUMNS
        # =================================================

        disease_columns = [
            "healthy",
            "multiple_diseases",
            "rust",
            "scab"
        ]


        # =================================================
        # CLEAN DATA
        # =================================================

        data = df[disease_columns].copy()

        data = data.fillna(0)

        data = data.astype("float32")


        # =================================================
        # PREDICTION USING AUTOENCODER
        # =================================================

        reconstructed = model.predict(
            data,
            verbose=0
        )


        # =================================================
        # RECONSTRUCTION ERROR
        # =================================================

        reconstruction_error = np.mean(
            np.square(
                data.values - reconstructed
            ),
            axis=1
        )


        df["reconstruction_error"] = (
            reconstruction_error
        )


        # =================================================
        # PREDICTED DISEASE
        # =================================================

        df["predicted_disease"] = df[
            disease_columns
        ].idxmax(axis=1)


        # =================================================
        # DISPLAY DATASET
        # =================================================

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )


        # =================================================
        # DATASET INFORMATION
        # =================================================

        st.subheader("Dataset Information")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Total Images",
                len(df)
            )

        with col2:

            st.metric(
                "Disease Classes",
                4
            )


        # =================================================
        # DISEASE DISTRIBUTION
        # =================================================

        st.subheader(
            "Leaf Disease Distribution"
        )

        disease_count = df[
            "predicted_disease"
        ].value_counts()


        st.bar_chart(
            disease_count
        )


        # =================================================
        # MOST COMMON DISEASE
        # =================================================

        if len(disease_count) > 0:

            most_common = disease_count.idxmax()

            count = disease_count.max()

            st.success(
                f"Most common predicted class: "
                f"{most_common.replace('_', ' ').title()} "
                f"({count} images)"
            )


        # =================================================
        # SELECT IMAGE ID
        # =================================================

        st.subheader(
            "Check Individual Prediction"
        )

        selected_image = st.selectbox(
            "Select Image ID",
            df["image_id"].tolist()
        )


        selected_row = df[
            df["image_id"] == selected_image
        ].iloc[0]


        predicted_class = selected_row[
            "predicted_disease"
        ]


        st.write(
            f"**Image ID:** {selected_image}"
        )

        st.write(
            f"**Predicted Class:** "
            f"{predicted_class.replace('_', ' ').title()}"
        )

        st.write(
            f"**Reconstruction Error:** "
            f"{selected_row['reconstruction_error']:.6f}"
        )


        # =================================================
        # PROBABILITY CHART
        # =================================================

        st.subheader(
            "Disease Probabilities"
        )

        probabilities = pd.DataFrame(
            {
                "Disease": disease_columns,
                "Probability": [
                    selected_row[col]
                    for col in disease_columns
                ]
            }
        )

        probabilities = probabilities.set_index(
            "Disease"
        )

        st.bar_chart(
            probabilities
        )


        # =================================================
        # DOWNLOAD RESULTS
        # =================================================

        result_csv = df.to_csv(
            index=False
        )

        st.download_button(
            label="Download Prediction Results",
            data=result_csv,
            file_name="leaf_disease_results.csv",
            mime="text/csv"
        )


    except Exception as e:

        st.error(
            f"Error while processing the file: {e}"
        )
