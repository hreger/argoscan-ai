import mlflow
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import io
from contextlib import redirect_stdout

class MLflowCallback(Callback):
    def __init__(self, validation_data=None, log_model=True):
        super().__init__()
        self.validation_data = validation_data
        self.log_model = log_model
        
    def on_train_begin(self, logs=None):
        """Log model architecture at the start of training."""
        # Get model summary
        model_summary = io.StringIO()
        with redirect_stdout(model_summary):
            self.model.summary()
        mlflow.log_text(model_summary.getvalue(), "model_summary.txt")
        
        # Log model parameters
        mlflow.log_params({
            'optimizer': self.model.optimizer.__class__.__name__,
            'learning_rate': float(self.model.optimizer.learning_rate.numpy()),
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        })
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch."""
        logs = logs or {}
        
        # Log training metrics
        mlflow.log_metrics({
            'train_loss': logs.get('loss'),
            'train_accuracy': logs.get('accuracy'),
            'val_loss': logs.get('val_loss'),
            'val_accuracy': logs.get('val_accuracy')
        }, step=epoch)
        
        # Calculate and log additional metrics if validation data is available
        if self.validation_data:
            y_pred = np.argmax(self.model.predict(self.validation_data[0]), axis=1)
            y_true = np.argmax(self.validation_data[1], axis=1) if len(self.validation_data[1].shape) > 1 else self.validation_data[1]
            
            # Calculate precision, recall, and f1 score
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted'
            )
            
            # Log additional metrics
            mlflow.log_metrics({
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }, step=epoch)
    
    def on_train_end(self, logs=None):
        """Log and register the model at the end of training."""
        if self.log_model:
            # Save model architecture and weights
            mlflow.tensorflow.log_model(
                self.model,
                "model",
                registered_model_name="AgroScanAI"
            )
            
            # Log final metrics
            if self.validation_data:
                y_pred = np.argmax(self.model.predict(self.validation_data[0]), axis=1)
                y_true = np.argmax(self.validation_data[1], axis=1) if len(self.validation_data[1].shape) > 1 else self.validation_data[1]
                
                final_accuracy = accuracy_score(y_true, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='weighted'
                )
                
                mlflow.log_metrics({
                    'final_accuracy': final_accuracy,
                    'final_precision': precision,
                    'final_recall': recall,
                    'final_f1_score': f1
                })

def setup_mlflow(experiment_name="plant_disease_classification"):
    """Setup MLflow tracking with the specified experiment name."""
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(experiment_name)
    
    # Create MLflow callback
    def create_callback(validation_data=None, log_model=True):
        return MLflowCallback(validation_data=validation_data, log_model=log_model)
    
    return create_callback