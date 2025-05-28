import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Dict, List
from .data_loader import preprocess_image

class PlantDiseasePredictor:
    def __init__(self, model_path: str, class_names: List[str]):
        """Initialize the predictor with a trained model and class names.
        
        Args:
            model_path (str): Path to the trained model file
            class_names (List[str]): List of class names for prediction
        """
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names
    
    def predict_image(self, image: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Predict disease from preprocessed image.
        
        Args:
            image (np.ndarray): Preprocessed image array
            
        Returns:
            Tuple containing:
            - Predicted class name
            - Confidence score
            - Dictionary of all class probabilities
        """
        # Ensure image is in batch format
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Get predictions
        predictions = self.model.predict(image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get all class probabilities
        class_probabilities = {
            class_name: float(prob)
            for class_name, prob in zip(self.class_names, predictions[0])
        }
        
        return self.class_names[predicted_class_idx], confidence, class_probabilities
    
    def predict_from_file(self, image_path: str) -> Tuple[str, float, Dict[str, float]]:
        """Predict disease from image file.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Tuple containing:
            - Predicted class name
            - Confidence score
            - Dictionary of all class probabilities
        """
        # Preprocess image
        image = preprocess_image(image_path)
        
        # Get prediction
        return self.predict_image(image)
    
    def get_disease_info(self, disease_name: str) -> Dict[str, str]:
        """Get information about a plant disease.
        
        Args:
            disease_name (str): Name of the disease
            
        Returns:
            Dictionary containing disease information
        """
        # TODO: Expand this with a proper disease information database
        return {
            "description": f"Information about {disease_name}",
            "symptoms": "Common symptoms include...",
            "treatment": "Recommended treatments include...",
            "prevention": "Prevention measures include..."
        }