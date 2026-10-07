import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import os


# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)


# ============================================
# TITLE
# ============================================

st.title("🌿 Leaf Disease Detection")

st.write(
    "Plant Pathology 2020 - Deep Learning Mini Project"
)

st.write(
    "Upload the sample submission CSV to analyze "
    "leaf disease prediction data."
)


# ============================================
# MODEL PATH
# ============================================

MODEL_PATH = "leaf_disease_autoencoder.keras"


# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


model = load_model()


# ============================================
# CHECK MODEL
# ============================================

if model is None:

    st.error(
        "Model file not found. Please upload "
        "leaf_disease_autoencoder.keras to GitHub."
    )

    st.stop()


# ============================================
# CSV UPLOAD
# ============================================

uploaded_file = st.file_uploader(
    "Upload sample_submission CSV",
    type=["csv"]
)


# ============================================
# PROCESS FILE
# ============================================

if uploaded_file is not None:

    try:

        # Read CSV
        df = pd.read_csv(uploaded_file)


        # Required columns
        disease_columns = [
            "healthy",
            "multiple_diseases",
            "rust",
            "scab"
        ]


        # Check columns
        required_columns = [
            "image_id",
            "healthy",
            "multiple_diseases",
            "rust",
            "scab"
        ]


        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]


        if missing_columns:

            st.error(
                "Missing columns: "
                + ", ".join(missing_columns)
            )

            st.stop()


        # ========================================
        # DATA
        # ========================================

        data = df[disease_columns].copy()

        data = data.fillna(0)

        data = data.astype("float32")


        # ========================================
        # AUTOENCODER PREDICTION
        # ========================================

        reconstructed = model.predict(
            data,
            verbose=0
        )


        # ========================================
        # RECONSTRUCTION ERROR
        # ========================================

        reconstruction_error = np.mean(
            np.square(
                data.values - reconstructed
            ),
            axis=1
        )


        df["reconstruction_error"] = (
            reconstruction_error
        )


        # ========================================
        # DISEASE CLASS
        # ========================================

        df["predicted_disease"] = df[
            disease_columns
        ].idxmax(axis=1)


        # ========================================
        # DATASET PREVIEW
        # ========================================

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )


        # ========================================
        # DATASET DETAILS
        # ========================================

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


        # ========================================
        # DISEASE DISTRIBUTION
        # ========================================

        st.subheader(
            "Disease Distribution"
        )


        disease_count = (
            df["predicted_disease"]
            .value_counts()
        )


        st.bar_chart(
            disease_count
        )


        # ========================================
        # MOST COMMON DISEASE
        # ========================================

        if len(disease_count) > 0:

            most_common = (
                disease_count.idxmax()
            )

            count = disease_count.max()


            st.success(
                "Most common predicted class: "
                + most_common.replace(
                    "_", " "
                ).title()
                + f" ({count} images)"
            )


        # ========================================
        # SELECT IMAGE
        # ========================================

        st.subheader(
            "Individual Image Analysis"
        )


        selected_image = st.selectbox(
            "Select Image ID",
            df["image_id"].tolist()
        )


        selected_row = df[
            df["image_id"] == selected_image
        ].iloc[0]


        # ========================================
        # RESULT
        # ========================================

        predicted_class = (
            selected_row["predicted_disease"]
        )


        st.write(
            f"**Image ID:** {selected_image}"
        )


        st.write(
            "**Predicted Class:** "
            + predicted_class
            .replace("_", " ")
            .title()
        )


        st.write(
            "**Reconstruction Error:** "
            + f"{selected_row['reconstruction_error']:.6f}"
        )


        # ========================================
        # PROBABILITIES
        # ========================================

        st.subheader(
            "Disease Probabilities"
        )


        probability_data = pd.DataFrame({

            "Disease": disease_columns,

            "Probability": [
                float(
                    selected_row[column]
                )
                for column in disease_columns
            ]

        })


        probability_data = (
            probability_data
            .set_index("Disease")
        )


        st.bar_chart(
            probability_data
        )


        # ========================================
        # DOWNLOAD RESULTS
        # ========================================

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
            "Error while processing CSV: "
            + str(e)
        )
