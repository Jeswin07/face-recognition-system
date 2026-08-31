# End-to-End Face Detection and Recognition System

An end-to-end face detection and recognition system built using Python, OpenCV, InsightFace, SCRFD, and ArcFace.

The system supports:

- Face detection in images
- Multiple-face detection
- Face recognition using ArcFace embeddings
- Identity matching using cosine similarity
- Unknown-face rejection using a configurable threshold
- Small cropped-face recognition fallback
- Dataset preparation and train/test splitting
- Data augmentation
- Recognition evaluation
- Threshold analysis
- Streamlit web interface
- Automated unit and integration testing

---

## 1. Project Overview

The objective of this project is to build a complete face detection and recognition pipeline starting from dataset preparation and ending with an interactive recognition interface.

The pipeline is:

```text
Input Image
     |
     v
SCRFD Face Detection
     |
     v
Detected Face + Facial Landmarks
     |
     v
ArcFace Embedding
     |
     v
512-D Face Embedding
     |
     v
Cosine Similarity
     |
     v
Known Identity / Unknown
````

For very small images that are already cropped to a face, the system provides an additional fallback:

```text
Small Face Crop
     |
     v
ArcFace Embedding
     |
     v
Cosine Similarity
     |
     v
Identity / Unknown
```

---

# 2. Dataset

The project uses a dataset containing:

* 600 total images
* 50 identities
* 12 images per identity

The dataset is divided into:

| Split    | Images | Percentage |
| -------- | -----: | ---------: |
| Training |    480 |        80% |
| Testing  |    120 |        20% |
| Total    |    600 |       100% |

The train/test split is performed before recognition evaluation to ensure that the test images are kept separate from the training embeddings.

Dataset metadata is stored in:

```text
data/processed/dataset_manifest.csv
data/processed/train.csv
data/processed/test.csv
```

---

# 3. Data Preparation

The project contains utilities for:

* Dataset downloading
* Dataset validation
* Dataset organization
* Train/test splitting
* Dataset manifest generation
* Image augmentation

Relevant modules:

```text
src/data/
├── augment.py
├── build_dataset.py
├── config.py
├── dataset.py
├── download.py
├── split.py
├── test_augmentation.py
├── torch_dataset.py
└── validate.py
```

---

# 4. Data Augmentation

Training images can be augmented to improve robustness and generalization.

The augmentation pipeline includes transformations such as:

* Rotation
* Scaling
* Horizontal flipping
* Brightness variation
* Other image-level transformations implemented in the augmentation module

Augmentation examples and analysis are stored under:

```text
outputs/plots/augmentation_examples.png
```

The important principle is that augmentation is applied to the training data and not used to contaminate the test evaluation.

---

# 5. Face Detection

The system uses:

## SCRFD

SCRFD is used for deep-learning-based face detection through InsightFace.

The configured detector uses:

```text
Model: SCRFD / det_10g
Detection size: 640 × 640
Detection threshold: 0.50
```

The detector can identify multiple faces in the same image.

Each detection contains information such as:

```text
Bounding box
Detection confidence
Facial landmarks
ArcFace embedding
```

Example:

```text
Face 1 | detection confidence: 0.9233
Face 2 | detection confidence: 0.9033
...
```

---

# 6. Face Recognition

Recognition is performed using:

## ArcFace

The InsightFace Buffalo_L model provides a pretrained ArcFace recognition network.

The recognition model produces a 512-dimensional face embedding.

The embedding is compared against the stored training embeddings using cosine similarity.

The highest similarity identity is selected when the similarity exceeds the configured recognition threshold.

Otherwise, the face is classified as:

```text
Unknown
```

Current default threshold:

```text
0.45
```

The threshold can be changed through the Streamlit interface.

---

# 7. Recognition Database

Training embeddings are stored in:

```text
data/embeddings/train_embeddings.npz
```

The project contains:

```text
480 training embeddings
50 identities
```

Identity information is also available through:

```text
data/embeddings/label_mapping.csv
```

Test embeddings are stored in:

```text
data/embeddings/test_embeddings.npz
```

---

# 8. Recognition Evaluation

The recognition system was evaluated on:

```text
Training images: 480
Test images:     120
Identities:       50
```

Using a recognition threshold of:

```text
0.45
```

the current evaluation result is:

```text
Accuracy: 97.50%
Correct predictions: 117 / 120
```

The evaluation can be reproduced with:

```bash
uv run python -m src.recognition.test_recognizer
```

Example output:

```text
Train embeddings : 480
Test embeddings  : 120
Identities       : 50
Threshold        : 0.45

Accuracy         : 0.9750
Accuracy         : 97.50%
```

---

# 9. Threshold Analysis

Recognition threshold selection was also evaluated using similarity distributions and threshold experiments.

Generated analysis files include:

```text
outputs/plots/accuracy_vs_threshold.png
outputs/plots/similarity_distribution.png
outputs/plots/threshold_results.csv
outputs/plots/per_identity_accuracy.csv
```

The threshold controls the trade-off between:

* Accepting genuine identities
* Rejecting unknown identities

A lower threshold generally accepts more matches but can increase false positives.

A higher threshold is more conservative but can reject genuine faces.

---

# 10. End-to-End Pipeline

The complete command-line pipeline is:

```bash
uv run python -m src.pipeline.recognize_image
```

The pipeline:

1. Loads the input image
2. Initializes SCRFD
3. Detects faces
4. Extracts ArcFace embeddings
5. Compares embeddings against the training database
6. Assigns identities
7. Rejects low-similarity matches as `Unknown`
8. Draws bounding boxes and labels
9. Saves the annotated result

The default input image is:

```text
data/input/test.jpg
```

The default output is:

```text
outputs/results/recognized_test.jpg
```

---

# 11. Small Image Fallback

Very small images may not provide enough resolution for SCRFD to detect a face.

For this situation, the pipeline checks the image dimensions.

If the image is sufficiently small and SCRFD detects no face, it is treated as an already-cropped face.

Instead of:

```text
Image → Detection → Embedding
```

the fallback uses:

```text
Image → ArcFace Embedding
```

This allows small face crops to still be recognized.

For example:

```text
Input:
125 × 94 pixels

SCRFD:
0 faces detected

Fallback:
ArcFace embedding

Result:
Andy Roddick
Similarity: 0.4813
```

This fallback is also covered by automated tests.

---

# 12. Streamlit User Interface

The project includes an interactive Streamlit application.

Start it with:

```bash
uv run streamlit run app.py
```

The interface provides:

* Image upload
* Original image preview
* Recognition result preview
* Face count
* Recognized face count
* Unknown face count
* Recognition similarity
* Detection confidence
* Configurable recognition threshold

The interface displays the complete recognition pipeline:

```text
Input Image
    ↓
SCRFD Detection
    ↓
ArcFace Embedding
    ↓
Cosine Similarity
    ↓
Identity / Unknown
```

---

# 13. Example Recognition Output

For a multi-face image, the system can produce results such as:

```text
Face  1 | Unknown         | similarity: 0.1590 | det: 0.9233
Face  2 | Unknown         | similarity: 0.1699 | det: 0.9033
Face  3 | Unknown         | similarity: 0.1518 | det: 0.8998
...
```

For a cropped face:

```text
Faces detected: 0

Small image detected.
Treating input as an already-cropped face.

Recognition: Andy Roddick
Similarity: 0.4813
```

---

# 14. Testing

The project includes automated tests for:

* Face detection
* Recognition evaluation
* Recognition utility functions
* Embedding loading
* Identity matching
* Unknown-face rejection
* End-to-end image processing
* Small-image fallback

Run all tests with:

```bash
uv run pytest
```

Current test result:

```text
18 passed
1 warning
```

Example:

```text
============================= test session starts ============================

collected 18 items

tests/test_detector.py ......        [ 33%]
tests/test_pipeline.py ..            [ 44%]
tests/test_recognizer.py .....       [ 72%]
tests/test_recognizer_unit.py .....  [100%]

====================== 18 passed, 1 warning in 5.78s =========================
```

The remaining warning comes from a deprecated function inside the InsightFace dependency:

```text
insightface/utils/face_align.py
```

It does not cause a test failure.

---

# 15. Project Structure

```text
face-recognition-system/
│
├── app.py
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
│
├── data/
│   ├── embeddings/
│   │   ├── label_mapping.csv
│   │   ├── train_embeddings.npz
│   │   └── test_embeddings.npz
│   │
│   ├── input/
│   │   ├── grppictestBM.jpg
│   │   ├── test.jpg
│   │   └── test_ui.jpg
│   │
│   └── processed/
│       ├── dataset_manifest.csv
│       ├── train.csv
│       └── test.csv
│
├── outputs/
│   ├── detection/
│   ├── plots/
│   └── results/
│
├── src/
│   ├── data/
│   │   ├── augment.py
│   │   ├── build_dataset.py
│   │   ├── config.py
│   │   ├── dataset.py
│   │   ├── download.py
│   │   ├── split.py
│   │   ├── torch_dataset.py
│   │   └── validate.py
│   │
│   ├── detection/
│   │   ├── detector.py
│   │   └── visualize.py
│   │
│   ├── pipeline/
│   │   └── recognize_image.py
│   │
│   ├── recognition/
│   │   ├── baseline.py
│   │   ├── embedder.py
│   │   ├── evaluate.py
│   │   ├── extract_embeddings.py
│   │   └── recognizer.py
│   │
│   ├── training/
│   │
│   └── utils/
│       └── device.py
│
└── tests/
    ├── test_detector.py
    ├── test_pipeline.py
    ├── test_recognizer.py
    └── test_recognizer_unit.py
```

---

# 16. Installation

This project uses `uv` for Python environment and dependency management.

Create/install the environment:

```bash
uv sync
```

Install development dependencies if necessary:

```bash
uv add --dev pytest
```

Run tests:

```bash
uv run pytest
```

Run the command-line pipeline:

```bash
uv run python -m src.pipeline.recognize_image
```

Run the Streamlit application:

```bash
uv run streamlit run app.py
```

---

# 17. Technologies Used

| Technology   | Purpose                            |
| ------------ | ---------------------------------- |
| Python 3.11  | Main programming language          |
| OpenCV       | Image processing and visualization |
| InsightFace  | Face detection and recognition     |
| SCRFD        | Face detection                     |
| ArcFace      | Face embeddings                    |
| ONNX Runtime | Model inference                    |
| NumPy        | Numerical operations               |
| Streamlit    | Interactive UI                     |
| Pytest       | Automated testing                  |
| uv           | Dependency/environment management  |

---

# 18. Assignment Requirement Mapping

## Phase 1 — Data Collection and Preparation

### Dataset with at least 500 images

Completed.

```text
600 images
50 identities
```

### 80/20 train/test split

Completed.

```text
480 training images
120 testing images
```

### Data augmentation

Completed.

Implemented in:

```text
src/data/augment.py
```

---

## Phase 2 — Face Detection

### Deep-learning face detector

Completed.

SCRFD from InsightFace is used.

### Multiple faces

Completed.

The detector supports multiple faces in a single image.

---

## Phase 3 — Face Recognition

### Deep-learning recognition model

Completed using the pretrained ArcFace recognition model supplied by InsightFace.

### Identity matching

Completed using cosine similarity against stored training embeddings.

### Evaluation

Completed.

Current measured accuracy:

```text
97.50%
117 / 120 correct
```

---

## Phase 4 — User Interface and Evaluation

### Image input UI

Completed using Streamlit.

### Detection and recognition in UI

Completed.

### Accuracy/evaluation plotting

Completed for threshold and similarity analysis.

Generated plots include:

```text
accuracy_vs_threshold.png
similarity_distribution.png
per_identity_accuracy.csv
threshold_results.csv
```

---

# 19. Optional Features

The project already contains InsightFace's age/gender model as part of the Buffalo_L model package, but age/gender estimation is not currently exposed as a primary application feature.

The following optional features are not implemented:

* Face tracking
* Video-stream tracking
* Liveness detection
* Anti-spoofing

These are optional bonus requirements and are not necessary for the core system.

---

# 20. Current Performance

Current recognition evaluation:

```text
Total identities : 50
Training images  : 480
Test images      : 120

Accuracy         : 97.50%
Correct          : 117 / 120
```

The system also successfully handles:

```text
✓ Single-face images
✓ Multiple-face images
✓ Unknown identities
✓ Low-confidence recognition
✓ Small cropped face images
✓ Configurable recognition threshold
```

---

# 21. Limitations

The recognition system uses pretrained ArcFace embeddings rather than training the ArcFace neural network from scratch.

This approach is intentional because it provides a strong face-recognition representation while allowing the project to focus on:

* Dataset preparation
* Detection
* Embedding extraction
* Identity matching
* Threshold selection
* Evaluation
* Application integration

Recognition performance can also depend on:

* Image resolution
* Face pose
* Lighting
* Occlusion
* Facial crop quality
* Similarity threshold

Very small images may require the crop fallback because a face detector may not have sufficient visual information to produce a reliable detection.

---

# 22. Final Result

The project provides a complete working image-based face recognition system with:

```text
Dataset
   ↓
Data Preparation
   ↓
Train/Test Split
   ↓
Augmentation
   ↓
SCRFD Face Detection
   ↓
ArcFace Embedding Extraction
   ↓
Cosine Similarity Matching
   ↓
Identity / Unknown
   ↓
Evaluation
   ↓
Streamlit UI
```

Final measured recognition accuracy:

## 97.50% (117 / 120)

Automated test suite:

## 18 / 18 tests passing

