# AgroScanAI - Plant Disease Detection

## Project Setup

### 1. Environment Setup
```bash
# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup
1. Download the PlantVillage dataset
2. Place the dataset in `data/PlantVillage/` directory
3. Initialize DVC and track the dataset:
```bash
# Initialize DVC (already done)
python -m dvc init

# Track the dataset
python -m dvc add data/PlantVillage

# Set up remote storage (optional)
python -m dvc remote add -d storage gdrive://<your-gdrive-folder-id>
```

### 3. DVC Pipeline
The project uses DVC to manage the data pipeline:

1. Data Preprocessing:
```bash
# Run preprocessing stage
python -m dvc repro preprocess
```

2. Model Training:
```bash
# Run training stage
python -m dvc repro train
```

### 4. Model Evaluation
```bash
# Run evaluation script
python -m app.evaluate
```

### 5. Streamlit App
```bash
# Run the web interface
streamlit run app/streamlit_app.py
```

## Project Structure
```
.
├── app/                    # Application code
│   ├── data_loader.py     # Data loading utilities
│   ├── evaluate.py        # Model evaluation
│   ├── gradcam.py         # Model interpretation
│   ├── model_builder.py   # Model architecture
│   ├── preprocess_data.py # Data preprocessing
│   └── streamlit_app.py   # Web interface
├── data/                  # Data directory
│   ├── PlantVillage/      # Raw dataset
│   └── processed_PlantVillage/ # Processed dataset
├── models/                # Model checkpoints
├── dvc.yaml              # DVC pipeline configuration
├── params.yaml           # Training parameters
└── requirements.txt      # Project dependencies
```

## DVC Pipeline Stages
1. **Preprocess**: Resizes and normalizes images to 128x128
2. **Train**: Trains MobileNetV2 model with transfer learning

## Model Architecture
- Base: MobileNetV2
- Input: 128x128x3 RGB images
- Output: Plant disease classification

## Performance Tracking
- MLflow for experiment tracking
- DVC for data and pipeline versioning
- Model metrics and artifacts stored in `metrics/`

