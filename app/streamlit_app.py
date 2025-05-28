import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from typing import Tuple

from .predict import PlantDiseasePredictor
from .explain import GradCAMExplainer
from .data_loader import preprocess_image

# Constants
MODEL_PATH = Path('models/latest_model.h5')
CLASS_NAMES = [  # TODO: Load from a config file
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Target_Spot',
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_Tomato_mosaic_virus',
    'Tomato_healthy'
]

def load_image(image_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess image for prediction.
    
    Args:
        image_file: Uploaded image file
        
    Returns:
        Tuple of original image array and preprocessed image array
    """
    # Read image
    image = Image.open(image_file)
    image_array = np.array(image)
    
    # Preprocess for model
    preprocessed = preprocess_image(image_file)
    
    return image_array, preprocessed

def main():
    st.set_page_config(
        page_title="AgroScanAI - Plant Disease Detection",
        page_icon="🌿",
        layout="wide"
    )
    
    # Add custom CSS
    st.markdown("""
        <style>
        .main {max-width: 1200px; margin: 0 auto; padding: 2rem;}
        .stTitle {color: #2e7d32;}
        .upload-box {border: 2px dashed #2e7d32; padding: 2rem; text-align: center;}
        .prediction-box {background-color: #f0f7f0; padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0;}
        .confidence-bar {height: 20px; background: linear-gradient(to right, #e8f5e9, #2e7d32); border-radius: 10px;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🌿 AgroScanAI - Plant Disease Detection")
    st.write("""
    Upload an image of a plant leaf to detect diseases and get detailed explanations.
    Our AI model will analyze the image and provide insights about plant health.
    """)
    
    # Initialize predictor and explainer
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        predictor = PlantDiseasePredictor(str(MODEL_PATH), CLASS_NAMES)
        explainer = GradCAMExplainer(model)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
            
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a single leaf for analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Create three columns for layout
            col1, col2, col3 = st.columns(3)
            
            # Display original image
            with col1:
                st.subheader("Original Image")
                image_array, preprocessed = load_image(uploaded_file)
                st.image(image_array, use_column_width=True)
            
            # Make prediction and display results
            prediction, confidence, class_probs = predictor.predict_image(preprocessed)
            
            with col2:
                st.subheader("Diagnosis Results")
                
                # Display prediction in styled box
                st.markdown(f"""
                    <div class='prediction-box'>
                        <h3 style='color: #2e7d32; margin-top: 0;'>Detected Condition</h3>
                        <p style='font-size: 1.2rem; font-weight: bold;'>{prediction.replace('_', ' ')}</p>
                        <p>Confidence: {confidence:.1%}</p>
                        <div class='confidence-bar' style='width: {confidence*100}%'></div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Disease information
                disease_info = predictor.get_disease_info(prediction)
                with st.expander("Disease Information", expanded=True):
                    st.write(f"**Description:** {disease_info['description']}")
                    st.write(f"**Symptoms:** {disease_info['symptoms']}")
                    st.write(f"**Treatment:** {disease_info['treatment']}")
                    st.write(f"**Prevention:** {disease_info['prevention']}")
                
                # Show all predictions
                with st.expander("All Predictions"):
                    for class_name, prob in class_probs.items():
                        st.write(f"{class_name.replace('_', ' ')}: {prob:.1%}")
            
            # Generate and display Grad-CAM visualization
            with col3:
                st.subheader("Model Explanation")
                class_idx = CLASS_NAMES.index(prediction)
                _, heatmap, overlay = explainer.explain_prediction(uploaded_file, class_idx)
                st.image(overlay, caption="Grad-CAM Visualization", use_column_width=True)
                st.write("""
                The highlighted regions show which parts of the image were most important
                for the model's prediction. Warmer colors (red) indicate stronger influence
                on the diagnosis.
                """)
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    # Project information
    st.markdown("---")
    st.markdown("""
    ### About AgroScanAI
    
    This tool uses advanced deep learning to detect plant diseases from leaf images.
    Features:
    - Accurate disease detection for tomato plants
    - Confidence scores for predictions
    - Visual explanations using Grad-CAM
    - Detailed disease information and treatment recommendations
    
    The model has been trained on the PlantVillage dataset and uses state-of-the-art
    techniques for both prediction and visualization.
    """)

if __name__ == "__main__":
    main()