# American Sign Language (ASL) Recognition System — Deep Learning & Django Web App

Final-year BSc (Hons) Computing software engineering production project implementing **American Sign Language (ASL) alphabet recognition** using deep learning models served through a **Django** web interface.

---

## Architecture & Highlights

- **Deep Learning & Computer Vision (` Implementation/Model Training/`)**
  - **Landmark Sequence Modeling (`asl_landmark_lstm_model.py`):** Extracts hand keypoints using **MediaPipe Hands**, applies image augmentation (rotation & brightness), and trains **TensorFlow / Keras** sequence models.
  - **Hybrid CNN-LSTM-Attention Model (`asl_mnist_cnn_lstm_attention_model.keras`):** Trained deep neural architecture for ASL sign classification.
  - **Rigorous Evaluation (`scikit-learn`, `matplotlib`):** Evaluates models using classification reports, confusion matrices, ROC-AUC curves, PR-AUC, Matthews Correlation Coefficient (MCC), Cohen's Kappa, log loss, and probability calibration curves.
- **Web Application (` Implementation/WebApp Source Code/ASl_UI/`)**
  - **Django** backend (`accounts/`, `asl/`, `mysite/`, `models/`) with user authentication, image/video upload handling (`media/`, `asl_images/`), and SQLite persistence.
- **Project Management & Engineering Lifecycle**
  - Full delivery artifacts across `Executing/`, `Monitoring and Controlling/`, `Portfolio/`, and `Meeting Record Sheets/`.

---

## Tech Stack

- **AI / ML / Vision:** Python, TensorFlow, Keras, MediaPipe, OpenCV (`cv2`), NumPy, scikit-learn, Matplotlib
- **Web Framework:** Django, HTML5, CSS3, SQLite
