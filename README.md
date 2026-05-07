# 🌿 AI Plant Pathology System

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)]()
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)]()

An automated, industry-grade computer vision pipeline designed to classify plant diseases from leaf imagery. This project leverages Transfer Learning to achieve **~97% validation accuracy** and features a lightweight, serverless-friendly web interface.

---

## ✨ Key Features
* **High-Accuracy Inference:** Fine-tuned `MobileNetV2` architecture optimized for edge-case leaf textures and spot patterns.
* **Smart Confidence Thresholding:** Built-in logic gates to reject anomalies, blurry photos, or out-of-distribution uploads (e.g., unsupported plant varieties) instead of forcing false-positive diagnoses.
* **Optimized ETL Pipeline:** Utilizes the `tf.data` API with prefetching and GPU-accelerated geometric data augmentation to prevent I/O bottlenecking during training.
* **Serverless UI:** A seamless, single-page application built with Streamlit for rapid deployment and easy user interaction.

---

## 🚀 Technical Architecture

### 1. The Core Engine
* **Base Model:** MobileNetV2 (Pre-trained on ImageNet).
* **Custom Head:** Global Average Pooling layer followed by Dropout (0.2) to prevent overfitting, terminating in a Softmax dense layer.
* **Optimization:** `Adam` optimizer initialized with a micro-learning rate ($10^{-5}$) for fine-tuning, managed dynamically by `ReduceLROnPlateau`.

### 2. Data Pipeline
* Images are dynamically loaded and processed in batches of 64.
* On-the-fly GPU augmentation includes random flips, 20° rotations, and 20% zoom variance to ensure robust feature extraction and prevent model memorization.

---

## 🛠️ Local Installation & Setup

It is highly recommended to run this project inside a virtual environment to prevent dependency conflicts.

**1. Clone the repository**
```bash
git clone [https://github.com/mrshivamroy/Plant-Doctor.git](https://github.com/mrshivamroy/Plant-Doctor.git)
cd Plant-Doctor
```
**2. Create and activate a virtual environment**
* Windows
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```
* macOS/Linux:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
**3. Install dependencies**
```bash
pip install -r requirements.txt
```
**4. Launch the application**
```
streamlit run app.py
```
The UI will automatically open in your default browser at http://localhost:8501.

## 📊 Model Performance
* **Training Accuracy:** ~96.4%
* **Validation Accuracy:** ~96.9%
* **Loss Function:** Categorical Crossentropy

Note: The model weights are compressed into the .keras format to optimize for deployment constraints without sacrificing predictive power.

## 🔮 Future Integrations
While currently monolithic using Streamlit, the model is designed to be easily decoupled. Future iterations will:

* Wrap the `.keras`model in a **FastAPI** or **Flask** microservice.

* Serve predictions via REST API to a custom full-stack client built with `Next.js` or a **MERN stack** architecture.

* Implement a database layer to track diagnosis history and geographic disease hotspots.

Designed & Developed by Shivam Roy