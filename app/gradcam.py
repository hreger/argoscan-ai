import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import cv2

class GradCAM:
    def __init__(self, model, layer_name=None):
        """Initialize GradCAM with a model and target layer name."""
        self.model = model
        
        # If layer_name is not provided, use the last conv layer
        if layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    layer_name = layer.name
                    break
        
        self.layer_name = layer_name
        self.grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )
    
    def compute_heatmap(self, image, class_idx=None, eps=1e-8):
        """Generate Grad-CAM heatmap for the given image and class."""
        with tf.GradientTape() as tape:
            conv_output, predictions = self.grad_model(image)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            class_output = predictions[:, class_idx]
        
        # Compute gradients of the target class with respect to the conv output
        grads = tape.gradient(class_output, conv_output)
        
        # Compute guided gradients
        cast_conv_output = tf.cast(conv_output > 0, tf.float32)
        cast_grads = tf.cast(grads > 0, tf.float32)
        guided_grads = cast_conv_output * cast_grads * grads
        
        # Average gradients spatially
        weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))
        
        # Build weighted feature map
        cam = tf.reduce_sum(tf.multiply(weights, conv_output), axis=-1)
        
        # Normalize heatmap
        heatmap = tf.maximum(cam, 0) / (tf.reduce_max(cam) + eps)
        heatmap = heatmap.numpy()
        
        return heatmap, predictions[0][class_idx].numpy()

def preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess image for model input."""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array, img

def generate_gradcam_visualization(image_path, model_path, class_names, output_path=None):
    """Generate and save Grad-CAM visualizations for a given image."""
    # Load model and create GradCAM instance
    model = load_model(model_path)
    gradcam = GradCAM(model)
    
    # Load and preprocess image
    img_array, original_img = preprocess_image(image_path)
    
    # Generate heatmap
    heatmap, confidence = gradcam.compute_heatmap(img_array)
    
    # Convert heatmap to RGB
    heatmap = cv2.resize(heatmap, (128, 128))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert original image to RGB numpy array
    original_img = img_to_array(original_img)
    original_img = np.uint8(original_img)
    
    # Create superimposed image
    superimposed = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    
    # Get predicted class
    pred_class = tf.argmax(model.predict(img_array)[0]).numpy()
    class_name = class_names[pred_class]
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    ax1.imshow(original_img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot heatmap
    ax2.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    ax2.set_title('Grad-CAM Heatmap')
    ax2.axis('off')
    
    # Plot superimposed image
    ax3.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    ax3.set_title(f'Overlay\n{class_name}\nConfidence: {confidence:.2%}')
    ax3.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        return output_path
    else:
        return plt.gcf()

if __name__ == '__main__':
    # Example usage
    model_path = 'models/latest_model.h5'
    image_path = 'data/PlantVillage/test_image.jpg'  # Replace with actual test image
    output_path = 'models/gradcam_visualization.png'
    
    # Load class names from data loader
    from data_loader import load_plantvillage_dataset, get_dataset_info
    _, val_dataset = load_plantvillage_dataset('data/PlantVillage', (128, 128), 32, 0.2)
    class_names = get_dataset_info(val_dataset)['class_names']
    
    # Generate visualization
    generate_gradcam_visualization(image_path, model_path, class_names, output_path)