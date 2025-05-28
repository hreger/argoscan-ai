import os
import tensorflow as tf
import tf2onnx
import onnx

def convert_keras_to_onnx(model_path, output_path):
    """Convert Keras model to ONNX format.
    
    Args:
        model_path (str): Path to the Keras model (.h5 file)
        output_path (str): Path to save the ONNX model
    """
    # Load the Keras model
    model = tf.keras.models.load_model(model_path)
    
    # Convert the model to ONNX
    input_signature = [tf.TensorSpec([None, 128, 128, 3], tf.float32, name='input_image')]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature)
    
    # Save the ONNX model
    onnx.save_model(onnx_model, output_path)
    print(f"Model converted and saved to {output_path}")

def main():
    # Define paths
    keras_model_path = os.path.join('models', 'latest_model.h5')
    onnx_model_path = os.path.join('models', 'plant_disease_model.onnx')
    
    # Check if Keras model exists
    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"Keras model not found at {keras_model_path}")
    
    # Convert model
    convert_keras_to_onnx(keras_model_path, onnx_model_path)

if __name__ == "__main__":
    main()