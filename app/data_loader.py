import tensorflow as tf
import os

def load_plantvillage_dataset(data_dir, img_size=(128, 128), batch_size=32, validation_split=0.2):
    """
    Load and preprocess the PlantVillage dataset using TensorFlow's image_dataset_from_directory.
    
    Args:
        data_dir (str): Path to the dataset directory
        img_size (tuple): Target size for image resizing
        batch_size (int): Number of samples per batch
        validation_split (float): Fraction of data to reserve for validation
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    
    # Data augmentation for training
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])
    
    # Normalization layer
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    # Load training dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset='training',
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    
    # Load validation dataset
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset='validation',
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    
    # Configure datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_dataset = train_dataset.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=AUTOTUNE
    ).map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)
    
    val_dataset = val_dataset.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)
    
    return train_dataset, val_dataset

def get_class_names(data_dir):
    """
    Get the list of class names from the dataset directory.
    
    Args:
        data_dir (str): Path to the dataset directory
        
    Returns:
        list: List of class names
    """
    return sorted(os.listdir(data_dir))

def get_dataset_info(dataset):
    """
    Get basic information about the dataset.
    
    Args:
        dataset: TensorFlow dataset object
        
    Returns:
        dict: Dictionary containing dataset information
    """
    info = {
        'num_classes': len(dataset.class_names),
        'class_names': dataset.class_names,
    }
    
    # Count total number of samples
    total_samples = 0
    for _ in dataset:
        batch_size = _.shape[0]
        total_samples += batch_size
    info['total_samples'] = total_samples
    
    return info