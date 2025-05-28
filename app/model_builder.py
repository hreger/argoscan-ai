import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def build_mobilenetv2_classifier(num_classes=10, input_shape=(128, 128, 3)):
    """
    Build a CNN classifier using MobileNetV2 as base model with custom classifier head.
    
    Args:
        num_classes (int): Number of output classes
        input_shape (tuple): Input image dimensions (height, width, channels)
        
    Returns:
        tf.keras.Model: Compiled model ready for training
    """
    # Load MobileNetV2 base model
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom classifier head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def unfreeze_top_layers(model, num_layers=30):
    """
    Unfreeze the top layers of the base model for fine-tuning.
    
    Args:
        model (tf.keras.Model): The model to fine-tune
        num_layers (int): Number of top layers to unfreeze
    """
    base_model = model.layers[1]  # MobileNetV2 is the second layer
    base_model.trainable = True
    
    # Freeze all layers except the top num_layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
        
    # Recompile the model with a lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_model_summary(model):
    """
    Get a string representation of the model architecture.
    
    Args:
        model (tf.keras.Model): The model to summarize
        
    Returns:
        str: Model summary string
    """
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    return '\n'.join(model_summary)