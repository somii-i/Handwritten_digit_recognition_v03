import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from data.preprocess import load_and_preprocess_data, get_train_val_split

def evaluate_model(model, X_val, Y_val):
    """Evaluate model performance with detailed metrics"""
    print("\n=== Starting Evaluation ===")
    print(f"Input shapes - X_val: {X_val.shape}, Y_val: {Y_val.shape}")
    
    results = model.evaluate(X_val, Y_val, verbose=0)
    print("\n=== Basic Metrics ===")
    print(f"Loss: {results[0]:.4f}")
    print(f"Accuracy: {results[1]*100:.2f}%")
    
    # Detailed classification report
    y_pred = model.predict(X_val, verbose=0)
    y_true = np.argmax(Y_val, axis=1)
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred_classes))
    
    return results

def analyze_predictions(model, test_data, num_samples=100):
    """Analyze and visualize prediction distribution"""
    print("\n=== Making Predictions ===")
    print(f"Test data shape: {test_data.shape}")
    
    predictions = model.predict(test_data[:num_samples], verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    # Prediction distribution
    unique, counts = np.unique(pred_classes, return_counts=True)
    print("\n=== Prediction Distribution ===")
    for digit, count in zip(unique, counts):
        print(f"Digit {digit}: {count} samples")
    
    # Visualize predictions
    plt.figure(figsize=(12,6))
    plt.bar(unique, counts)
    plt.title("Prediction Distribution")
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.xticks(range(10))
    plt.show()
    
    return pred_classes

def plot_confusion_matrix(model, X_val, Y_val):
    """Plot detailed confusion matrix"""
    y_pred = model.predict(X_val, verbose=0)
    y_true = np.argmax(Y_val, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

if __name__ == '__main__':
    try:
        # Load best saved model
        model = load_model(Config.MODEL_SAVE_PATH)
        print(f"Successfully loaded model from {Config.MODEL_SAVE_PATH}")
        
        # Load and prepare data
        X_train, Y_train, test_data = load_and_preprocess_data()
        X_train, X_val, Y_train, Y_val = get_train_val_split(X_train, Y_train)
        
        # Evaluate
        loss, acc = evaluate_model(model, X_val, Y_val)
        
        # Analyze predictions
        predictions = analyze_predictions(model, X_val)
        
        # Show confusion matrix
        plot_confusion_matrix(model, X_val, Y_val)
        
    except Exception as e:
        print(f"\n!!! Error: {str(e)}")
        print("Possible causes:")
        print("- Model not trained (run train.py first)")
        print("- Corrupted model file")
        print("- Data loading issues")