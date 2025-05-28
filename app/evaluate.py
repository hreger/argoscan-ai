import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
import tensorflow as tf
from data_loader import load_plantvillage_dataset, get_dataset_info

def predict_dataset(model, dataset):
    """Get predictions for the entire dataset."""
    y_true = []
    y_pred = []
    
    for images, labels in dataset:
        # Get predictions
        predictions = model.predict(images)
        
        # Convert one-hot encoded labels and predictions to class indices
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
    
    return np.array(y_true), np.array(y_pred)

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix using seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    plt.close()

def plot_per_class_accuracy(y_true, y_pred, class_names):
    """Plot per-class accuracy."""
    class_accuracies = []
    for i in range(len(class_names)):
        mask = y_true == i
        class_acc = accuracy_score(y_true[mask], y_pred[mask])
        class_accuracies.append(class_acc)
    
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, class_accuracies)
    plt.title('Per-Class Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('models/per_class_accuracy.png')
    plt.close()

def evaluate_model(model_path):
    """Evaluate the trained model and generate performance metrics and visualizations."""
    # Load the trained model
    model = load_model(model_path)
    
    # Load validation dataset
    _, val_dataset = load_plantvillage_dataset(
        data_dir='data/PlantVillage',
        img_size=(128, 128),
        batch_size=32,
        validation_split=0.2
    )
    
    # Get class names
    dataset_info = get_dataset_info(val_dataset)
    class_names = dataset_info['class_names']
    
    # Get predictions
    y_true, y_pred = predict_dataset(model, val_dataset)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Print classification report
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Print overall metrics
    print('\nOverall Metrics:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
    
    # Generate plots
    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_per_class_accuracy(y_true, y_pred, class_names)
    
    print('\nVisualization plots have been saved in the models directory.')

if __name__ == '__main__':
    model_path = 'models/latest_model.h5'
    evaluate_model(model_path)