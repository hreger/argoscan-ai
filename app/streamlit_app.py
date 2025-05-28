import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.models import load_model
from gradcam import generate_gradcam_visualization, preprocess_image
from data_loader import get_dataset_info, load_plantvillage_dataset

# Set page config
st.set_page_config(
    page_title="AgroScanAI - Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

def load_model_and_classes():
    """Load the trained model and class names."""
    model_path = os.path.join('models', 'latest_model.h5')
    model = load_model(model_path)
    
    # Get class names
    _, val_dataset = load_plantvillage_dataset(
        data_dir=os.path.join('data', 'PlantVillage'),
        img_size=(128, 128),
        batch_size=32,
        validation_split=0.2
    )
    class_names = get_dataset_info(val_dataset)['class_names']
    
    return model, class_names

def get_disease_description(class_name):
    """Return a description and treatment recommendation for the disease."""
    # This could be expanded with a proper database of disease information
    return {
        'description': f"Detected {class_name}. This is a common plant disease that affects crop health and yield.",
        'treatment': "General recommendations:\n" + \
                    "1. Remove affected leaves\n" + \
                    "2. Improve air circulation\n" + \
                    "3. Apply appropriate fungicide if necessary\n" + \
                    "4. Monitor plant health regularly"
    }

def main():
    # Add custom CSS
    st.markdown("""
        <style>
        .main {max-width: 1200px; margin: 0 auto; padding: 2rem;}
        .stTitle {color: #2e7d32;}
        .upload-box {border: 2px dashed #2e7d32; padding: 2rem; text-align: center;}
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title('🌿 AgroScanAI - Plant Disease Detection')
    st.markdown('''
        Upload a photo of a plant leaf to detect diseases and get treatment recommendations.
        Our AI model will analyze the image and provide detailed insights.
    ''')
    
    # Load model and classes
    try:
        model, class_names = load_model_and_classes()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a leaf image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a single leaf"
    )
    
    if uploaded_file is not None:
        try:
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            # Display original image
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
                
                # Save temporary file for Grad-CAM
                temp_path = os.path.join('models', 'temp_image.jpg')
                image.save(temp_path)
            
            # Process image and make prediction
            img_array, _ = preprocess_image(temp_path)
            prediction = model.predict(img_array)
            pred_class_idx = np.argmax(prediction[0])
            confidence = prediction[0][pred_class_idx]
            predicted_class = class_names[pred_class_idx]
            
            # Get disease information
            disease_info = get_disease_description(predicted_class)
            
            # Display prediction results
            with col2:
                st.subheader("Analysis Results")
                
                # Create a styled box for the prediction
                st.markdown(f"""
                    <div style='padding: 1rem; border-radius: 0.5rem; background-color: #f0f7f0;'>
                        <h3 style='color: #2e7d32;'>Diagnosis</h3>
                        <p><strong>Detected Condition:</strong> {predicted_class}</p>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display disease information
                st.markdown("### Description")
                st.write(disease_info['description'])
                
                st.markdown("### Treatment Recommendations")
                st.write(disease_info['treatment'])
            
            # Generate and display Grad-CAM visualization
            st.subheader("Model Interpretation (Grad-CAM)")
            gradcam_path = os.path.join('models', 'gradcam_output.png')
            generate_gradcam_visualization(
                temp_path,
                os.path.join('models', 'latest_model.h5'),
                class_names,
                gradcam_path
            )
            st.image(gradcam_path, use_column_width=True)
            
            # Cleanup temporary files
            os.remove(temp_path)
            os.remove(gradcam_path)
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    # Add information about the project
    st.markdown("---")
    st.markdown("""
        ### About AgroScanAI
        This tool uses deep learning to detect plant diseases from leaf images.
        The model has been trained on the PlantVillage dataset and uses Grad-CAM
        for visualization of important regions in the image that influenced the prediction.
    """)

if __name__ == "__main__":
    main()