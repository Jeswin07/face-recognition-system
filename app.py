from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.detection.detector import FaceDetector
from src.recognition.embedder import FaceEmbedder
from src.recognition.recognizer import FaceRecognizer
from src.pipeline.recognize_image import process_image


EMBEDDINGS_PATH = Path("data/embeddings/train_embeddings.npz")


@st.cache_resource
def load_models(threshold: float):
    detector = FaceDetector(
        det_size=(640, 640),
        det_thresh=0.5,
    )

    embedder = FaceEmbedder()

    recognizer = FaceRecognizer(
        EMBEDDINGS_PATH,
        threshold=threshold,
    )

    return detector, embedder, recognizer


# =============================================================
# STREAMLIT UI
# =============================================================

st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide",
)

st.title("👤 End-to-End Face Detection & Recognition")

st.caption(
    "InsightFace SCRFD + ArcFace | "
    "50 identities | 97.50% test accuracy"
)

# -------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------

st.sidebar.header("Settings")

threshold = st.sidebar.slider(
    "Recognition Threshold",
    min_value=0.20,
    max_value=0.80,
    value=0.45,
    step=0.01,
)

st.sidebar.info(
    "Faces with similarity below the threshold "
    "are classified as Unknown."
)

# -------------------------------------------------------------
# Upload
# -------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        st.error("Could not read the uploaded image.")
        st.stop()

    # ---------------------------------------------------------
    # Load models
    # ---------------------------------------------------------

    with st.spinner("Loading face recognition models..."):

        detector, embedder, recognizer = load_models(
            threshold
        )

    # ---------------------------------------------------------
    # Process
    # ---------------------------------------------------------

    with st.spinner(
        "Detecting and recognizing faces..."
    ):

        output, results, used_crop_fallback = process_image(
            image,
            detector,
            embedder,
            recognizer,
        )

    # ---------------------------------------------------------
    # Images
    # ---------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original")

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            ),
            use_container_width=True,
        )

    with col2:

        st.subheader("Recognition Result")

        st.image(
            cv2.cvtColor(
                output,
                cv2.COLOR_BGR2RGB,
            ),
            use_container_width=True,
        )

    st.divider()

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    st.subheader("Detection Results")

    if not results:

        st.warning(
            "No faces detected."
        )

    else:

        recognized = sum(
            result["identity"] != "Unknown"
            for result in results
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Faces Detected",
            len(results),
        )

        col2.metric(
            "Recognized",
            recognized,
        )

        col3.metric(
            "Unknown",
            len(results) - recognized,
        )

        # -----------------------------------------------------
        # Results table
        # -----------------------------------------------------

        display_results = []

        for result in results:

            display_results.append(
                {
                    "face": result["face"],
                    "identity": result["identity"],
                    "similarity": (
                        round(
                            result["similarity"],
                            4,
                        )
                    ),
                    "detection": (
                        round(
                            result["detection"],
                            4,
                        )
                        if result["detection"] is not None
                        else "N/A"
                    ),
                }
            )

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True,
        )

        # -----------------------------------------------------
        # Crop fallback information
        # -----------------------------------------------------

        if used_crop_fallback:

            st.success(
                "Small face crop recognized using "
                "direct ArcFace embedding."
            )

else:

    st.info(
        "Upload an image to detect and recognize faces."
    )

    st.markdown(
        """
        ### System Pipeline

        **Normal image**

        Input Image
        → SCRFD Face Detection
        → ArcFace 512-D Embedding
        → Cosine Similarity
        → Identity / Unknown

        **Small face crop**

        Input Crop
        → ArcFace 512-D Embedding
        → Cosine Similarity
        → Identity / Unknown

        ### Current Performance

        - Dataset: **600 images**
        - Identities: **50**
        - Training images: **480**
        - Test images: **120**
        - Recognition accuracy: **97.50%**
        - Correct predictions: **117 / 120**
        """
    )