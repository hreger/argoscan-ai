import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Tuple
from PIL import Image
from .data_loader import preprocess_image

class GradCAMExplainer:
    def __init__(self, model: tf.keras.Model):
        """Initialize GradCAM explainer.
        
        Args:
            model (tf.keras.Model): Trained model to explain
        """
        self.model = model
        # Get the last convolutional layer
        self.last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                self.last_conv_layer = layer
                break
        if self.last_conv_layer is None:
            raise ValueError("Could not find convolutional layer in model")
    
    def compute_heatmap(self, image: np.ndarray, class_idx: int) -> np.ndarray:
        """Compute Grad-CAM heatmap for a given image and class.
        
        Args:
            image (np.ndarray): Preprocessed image array
            class_idx (int): Index of the target class
            
        Returns:
            np.ndarray: Generated heatmap
        """
        # Create a model that maps the input image to the activations
        # of the last conv layer and the output predictions
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.last_conv_layer.output, self.model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            class_output = predictions[:, class_idx]
        
        # Extract gradients
        grads = tape.gradient(class_output, conv_outputs)
        
        # Global average pooling
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by corresponding gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(
            tf.multiply(pooled_grads, conv_outputs), axis=-1
        ).numpy()
        
        # Normalize the heatmap
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) or 1e-10)
        return heatmap
    
    def create_visualization(self, image: np.ndarray, heatmap: np.ndarray,
                           alpha: float = 0.4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create visualization with original image, heatmap, and overlay.
        
        Args:
            image (np.ndarray): Original image
            heatmap (np.ndarray): Generated heatmap
            alpha (float): Transparency factor for overlay
            
        Returns:
            Tuple containing:
            - Original image
            - Colored heatmap
            - Heatmap overlay on original image
        """
        # Resize heatmap to match image size
        heatmap = np.uint8(255 * heatmap)
        heatmap = tf.image.resize(heatmap[..., tf.newaxis], image.shape[:2],
                                method='bilinear').numpy()
        
        # Convert heatmap to RGB
        cmap = plt.get_cmap('jet')
        colored_heatmap = cmap(heatmap[..., 0])[..., :3] * 255
        
        # Create overlay
        overlay = colored_heatmap * alpha + image * (1 - alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return image, colored_heatmap.astype(np.uint8), overlay
    
    def explain_prediction(self, image_path: str, class_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate Grad-CAM visualization for an image.
        
        Args:
            image_path (str): Path to the image file
            class_idx (int): Index of the target class
            
        Returns:
            Tuple containing:
            - Original image
            - Colored heatmap
            - Heatmap overlay on original image
        """
        # Preprocess image
        image = preprocess_image(image_path)
        processed_image = np.expand_dims(image, axis=0)
        
        # Compute heatmap
        heatmap = self.compute_heatmap(processed_image, class_idx)
        
        # Create visualization
        return self.create_visualization(image, heatmap)