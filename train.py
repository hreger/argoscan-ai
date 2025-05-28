import os
import mlflow
import mlflow.tensorflow
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from app.data_loader import load_plantvillage_dataset, get_dataset_info
from app.model_builder import build_mobilenetv2_classifier, unfreeze_top_layers
from app.mlflow_tracking import setup_mlflow

def get_callbacks(checkpoint_path, validation_data):
    """Create training callbacks."""
    callbacks = [
        # Model checkpoint callback
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        # Early stopping callback
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate callback
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        # MLflow callback
        setup_mlflow()(validation_data=validation_data)
    ]
    return callbacks

def train_model(train_dataset, val_dataset, num_classes, epochs=20):
    """Train the model with the given datasets."""
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Build and compile model
        model = build_mobilenetv2_classifier(num_classes=num_classes)
        
        # Create checkpoint path
        checkpoint_path = os.path.join('models', 'checkpoints', 'model_{epoch:02d}_{val_accuracy:.4f}.h5')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Get validation data for metrics calculation
        validation_data = next(iter(val_dataset))
        
        # Get callbacks
        callbacks = get_callbacks(checkpoint_path, validation_data)
        
        # Initial training with frozen base model
        print("\nTraining with frozen base model...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Fine-tuning
        print("\nFine-tuning the model...")
        model = unfreeze_top_layers(model)
        
        # Update checkpoint path for fine-tuning
        checkpoint_path = os.path.join('models', 'checkpoints', 'fine_tuned_model_{epoch:02d}_{val_accuracy:.4f}.h5')
        callbacks = get_callbacks(checkpoint_path, validation_data)
        
        # Train the fine-tuned model
        history_fine = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=10,
            callbacks=callbacks
        )
        
        # Save the final model
        save_path = os.path.join('models', f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
        model.save(save_path)
        mlflow.log_artifact(save_path)
        
        # Save as latest model
        latest_model_path = os.path.join('models', 'latest_model.h5')
        model.save(latest_model_path)
        mlflow.log_artifact(latest_model_path)
        
        return model, history, history_fine

def main():
    # Load datasets
    data_dir = os.path.join('data', 'PlantVillage')
    train_dataset, val_dataset = load_plantvillage_dataset(
        data_dir=data_dir,
        img_size=(128, 128),
        batch_size=32,
        validation_split=0.2
    )
    
    # Get dataset info
    dataset_info = get_dataset_info(train_dataset)
    num_classes = dataset_info['num_classes']
    
    print(f"\nDataset Information:")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {dataset_info['class_names']}")
    print(f"Total training samples: {dataset_info['total_samples']}")
    
    # Train the model
    model, history, history_fine = train_model(
        train_dataset,
        val_dataset,
        num_classes=num_classes
    )
    
    print("\nTraining completed! Model saved in 'models' directory.")

if __name__ == "__main__":
    main()