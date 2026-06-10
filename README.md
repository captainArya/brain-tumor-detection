# 🧠 Explainable AI-Powered Brain Tumor Detection System

## Overview

This project is an intelligent Brain Tumor Detection and Analysis System that combines YOLOv8 object detection, Explainable AI (XAI), Memory-Based Similarity Search, and Gemini-powered AI report generation to assist in MRI-based brain tumor analysis.

The system detects tumors from MRI brain scans, identifies the tumor type, highlights the affected region, explains the prediction using visual explanations, and generates a detailed AI-assisted medical report.

---

## Features

### 🔍 Brain Tumor Detection

* Detects tumors from MRI brain scans using YOLOv8.
* Supports:

  * Glioma
  * Meningioma
  * Pituitary Tumor
  * No Tumor

### 📊 Confidence Scoring

* Displays model confidence score for each prediction.

### 🔥 Explainable AI (XAI)

* Highlights important tumor regions.
* Displays:

  * Tumor Location
  * Area Covered (%)
  * Confidence Score

### 🧠 Similar Case Retrieval

* Generates image embeddings.
* Finds similar cases from memory for reference.

### 🤖 AI Report Generation

* Uses Google Gemini API.
* Generates detailed medical reports including:

  * Clinical Summary
  * Tumor Description
  * Risk Assessment
  * Tumor Location Analysis
  * Area Coverage Interpretation
  * Recommendations
  * Patient-Friendly Explanation

### 📥 Report Download

* Download generated reports directly from the application.

### 💬 Feedback System

* Allows users to submit feedback for future improvements.

---

## Tech Stack

### Deep Learning

* YOLOv8
* PyTorch

### Computer Vision

* OpenCV
* Pillow

### Explainable AI

* Heatmap-based visual explanations
* Tumor localization

### Generative AI

* Google Gemini API

### Web Application

* Streamlit

### Data Processing

* NumPy
* Pandas

---

## Project Structure

```text
braintumor/
│
├── app.py
├── detect.py
├── report_generator.py
├── .env
├── uploads/
├── feedback/
├── feedback_dataset/
├── temp/
├── temp_results/
├── runs/
├── data/
│   ├── train/
│   ├── valid/
│   └── test/
└── README.md
```

---

## Workflow

1. Upload MRI Brain Scan
2. Run Tumor Detection
3. Predict Tumor Type
4. Generate Explainable AI Insights
5. Compute:

   * Confidence Score
   * Tumor Location
   * Area Covered
6. Generate AI Medical Report using Gemini
7. Download Report

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd braintumor
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Gemini API

Create a `.env` file:

```env
GEMINI_API_KEY=YOUR_API_KEY
```

### Run Application

```bash
streamlit run app.py
```

---

## Sample Output

* Tumor Type Detection
* Confidence Score
* Tumor Localization
* Explainable AI Heatmap
* AI Generated Medical Report

---

## Disclaimer

This project is intended for educational and research purposes only.

The generated reports are AI-assisted outputs and should not be considered a final medical diagnosis. Clinical decisions should always be made by qualified healthcare professionals.

---

## Author

Arya R B

AI / Machine Learning / Computer Vision Enthusiast
