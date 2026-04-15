
# Pre-Fall Detection System using MediaPipe and GRU

This repository contains the source code for a real-time, skeleton-based "loss-of-balance" (pre-fall) detection system. Unlike traditional post-fall detection systems that rely on impact heuristics, this pipeline utilizes spatial-temporal modeling to proactively identify instability. 

By leveraging Google's **MediaPipe** for lightweight pose estimation and a **Gated Recurrent Unit (GRU)** network for sequence classification, the system ensures privacy-preserving, camera-resolution-invariant monitoring at the edge.

---

## 🏗️ System Architecture

The application is structured as a 5-layer edge-computing pipeline:

1. **Video Ingestion Layer:** Captures real-time frames via OpenCV, optimized for minimal latency.
2. **Spatial Feature Extraction (MediaPipe):** Utilizes the `PoseLandmarker` Tasks API to extract 3D skeleton configurations. The system filters for 6 critical keypoints: Shoulders (11, 12), Hips (23, 24), and Knees (25, 26).
3. **Feature Engineering & Normalization:** * **Trigonometric Profiling:** Calculates dynamic left and right body-fold angles using `arctan2`.
   * **Scale Invariance:** Translates raw pixel coordinates into normalized space `[0.0, 1.0]`, ensuring the predictive model remains agnostic to varying camera focal lengths and resolutions.
4. **Temporal Classification (GRU):** Processes engineered features using a sliding window approach (`TIME_STEPS = 10`). The sequential data is fed into a multi-layer GRU neural network to capture the temporal dynamics of falling, outputting a binary probability state.
5. **Real-time Telemetry & UI:** Renders a non-blocking Heads-Up Display (HUD) using a custom `UIManager`, featuring dynamic risk-bars and automated background data logging for continuous model improvement.

---

## 📂 Repository Structure

```text
fall-detection-system/
├── core/                           # Core operational modules
│   ├── pose_estimator.py           # MediaPipe integration and landmark parsing
│   ├── angle_calculator.py         # Trigonometric joint angle computation
│   └── ui_manager.py               # HUD rendering and telemetry overlay
├── data/                           # Datasets and exploratory analysis outputs
│   ├── fall_dataset.csv            # Primary training dataset
│   ├── test_dataset.csv            # Unseen testing dataset
│   ├── live_collected_data.csv     # Auto-logged inference data
│   └── feature_statistics.csv      # EDA statistical outputs
├── assets/                         # Model weights and topologies
│   ├── pose_landmarker_full.task   # MediaPipe base weights
│   └── fall_model.keras            # Trained GRU network weights
├── collect_data.py                 # Script for recording raw human poses
├── analyze_features.py             # Exploratory Data Analysis (EDA) script
├── train_model.py                  # GRU model architecture and training loop
├── evaluate_model.py               # Model evaluation and metric generation
├── main.py                         # Primary inference engine (Real-time tracking)
└── requirements.txt                # Environment dependencies
```

---

## ⚙️ Environment Setup

**Prerequisites:** Python 3.8 - 3.11 is recommended.

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <your_repository_url>
   cd fall-detection-system
   ```
2. Initialize and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Machine Learning Pipeline & Usage

To replicate the project results or deploy the system, follow the operational pipeline in order:

### 1. Data Ingestion & Annotation (`collect_data.py`)
Used to generate sequential training and testing datasets.
* Run: `python collect_data.py`
* **Controls:** * Press `n` to record the Normal baseline state (Label: `0`).
  * Press `f` to record the Loss-of-Balance state (Label: `1`).
  * Press `p` (or Spacebar) to pause data collection.
  * Press `q` to exit and save the `.csv` tensor.

### 2. Exploratory Data Analysis (`analyze_features.py`)
Provides statistical explainability (Explainable AI) regarding which physiological features are most indicative of a fall.
* Run: `python analyze_features.py`
* Calculates the Mean and Standard Deviation grouped by class vectors, outputting an ordered matrix to `feature_statistics_report.csv`.

### 3. Neural Network Training (`train_model.py`)
Compiles and trains the GRU architecture.
* Run: `python train_model.py`
* **Mechanism:** Slices the dataset into sequences of 10 frames. The model trains for 30 Epochs (by default) to minimize binary cross-entropy loss. Outputs the finalized weights to `assets/fall_model.keras`.

### 4. Model Evaluation (`evaluate_model.py`)
Validates model robustness against unseen data to prevent overfitting.
* Run: `python evaluate_model.py`
* Generates a comprehensive Classification Report (Precision, Recall, F1-Score) and a Confusion Matrix detailing False Alarms and Missed Detections.

### 5. Edge Deployment / Real-time Inference (`main.py`)
Initializes the live monitoring system.
* Run: `python main.py`
* **Behavior:** The system will dynamically calculate pose states. If the probabilistic confidence of instability exceeds the predefined threshold (60%), the UI triggers a critical visual alert. All live telemetry is asynchronously logged to `live_collected_data.csv` for future data pipelines.

### Key Improvements Made:
1. **Removed Emojis & Student Phrasing:** Stripped away localized terms like "สมอง AI" (AI Brain) and replaced them with the correct technical counterparts (e.g., "GRU network weights").
2. **Clarified the "Why":** Rather than just saying "we convert coordinates to 0.0-1.0", the README now explains that this achieves *Scale Invariance*, proving technical competence.
3. **Professional Formatting:** Utilized bolding, code blocks, and standard Markdown tree structures to make the repository highly readable for other developers or potential employers checking your GitHub.