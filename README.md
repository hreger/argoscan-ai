##🌾 AgroScan AI – Smart Crop Monitoring & Disease Detection

AgroScan AI is a vision-based AI system that helps farmers detect crop diseases and plant anomalies using images. Built for accessibility and real-world deployment, AgroScan focuses on clear, actionable insights with a streamlined UI for non-technical users.

---

## 🚀 Phase 1: Disease Detection for Single Plants (Internship Scope)

### 🎯 Objective
Classify crop diseases in **single plant images** with visual explanations using deep learning. Deploy a minimal, user-friendly web app for farmers and agronomists.

---

## 🧠 Core Features

- ✅ **Crop Disease Classification** (MobileNetV2)
- ✅ **Visual Explanation with Grad-CAM** (Explainable AI)
- ✅ **Lightweight & Deployable via Streamlit**
- ✅ **MLflow for Experiment Tracking**
- ✅ **DVC for Dataset and Pipeline Versioning**

---

## 🔍 Sample Use Case

1. Farmer clicks a photo of a leaf or plant.
2. App predicts if it's healthy or diseased.
3. Visual overlay shows area of concern (e.g., leaf blight).
4. Simple suggestion/label is shown in regional language (future work).

---

## 🧱 Tech Stack

| Component           | Tool/Framework       |
|--------------------|----------------------|
| Classification     | MobileNetV2          |
| Explainability     | Grad-CAM              |
| UI                 | Streamlit            |
| Model Tracking     | MLflow                |
| Version Control    | Git + DVC             |
| Deployment         | Docker (Planned)      |

---

## 🖼️ System Architecture – Phase 1

```mermaid
flowchart TD
    User["User (Farmer)"]
    Upload["Upload Plant Image"]
    Classify["Disease Classification (MobileNetV2)"]
    GradCAM["Grad-CAM Heatmap"]
    Output["Prediction + Visual Explanation"]
    
    User --> Upload --> Classify --> GradCAM --> Output
````

---

## 📂 Directory Structure

```
agroscan-ai/
├── data/                 # Datasets (via DVC)
├── models/               # Saved model checkpoints
├── src/
│   ├── preprocessing.py  # Image preprocessing
│   ├── classify.py       # Classification logic
│   └── explain.py        # Grad-CAM visualization
├── app/
│   └── streamlit_app.py  # Streamlit UI
├── mlruns/               # MLflow experiments
├── dvc.yaml              # Pipeline tracking
└── README.md
```

---

## 📊 Model Evaluation

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 92.3% |
| Precision | 90.8% |
| Recall    | 91.4% |
| F1-score  | 91.1% |
| Classes   | 10    |

---

## ⚙️ How to Run Locally

```bash
# Clone Repo
git clone https://github.com/hreger/agroscan-ai
cd agroscan-ai

# Install Dependencies
pip install -r requirements.txt

# Run Streamlit App
streamlit run app/streamlit_app.py
```

---

## 🧪 Experiment Tracking (MLflow)

Launch MLflow UI:

```bash
mlflow ui
```

View results at: `http://localhost:5000`

---

## 🗃️ Dataset

* ✅ [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
* ✅ Augmented for lighting & rotation variance

---

## 🔐 Future Enhancements (Post-Internship)

| Feature                     | Description                                |
| --------------------------- | ------------------------------------------ |
| 🧠 Multi-plant Segmentation | YOLOv8-seg or Mask-RCNN for field images   |
| 📱 Mobile Deployment        | TensorRT/ONNX optimization + Flask backend |
| 🔄 Real-time Feedback       | Stream processing for video/camera input   |
| 🔒 User Security            | JWT auth + farmer login                    |
| 📡 Cloud Integration        | AWS Lambda/EC2 or GCP Functions            |

---

## 🌱 Project Vision

To empower farmers with an AI-powered mobile solution that delivers **actionable health diagnostics**, **farming advice**, and **resource optimization insights** using **explainable, field-tuned deep learning models**.

---

## 👩‍💻 Contributors

* P Sanjeev Pradeep
* Open to collaborations!

---

## 📄 License

MIT License. Free to use, modify, and build upon with credits.

