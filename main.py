import os
import sys
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import Config
from data.preprocess import load_and_preprocess_data, check_data_balance, get_train_val_split
from model.train import train_model
from model.evaluate import evaluate_model, analyze_predictions, plot_confusion_matrix

def main():
    try:
        print("\n=== Digit Recognition System ===")
        Config.create_dirs()
        
        # 1. Data Preparation
        print("\n[1/3] Loading and preprocessing data...")
        X_train, Y_train, test_data = load_and_preprocess_data()
        check_data_balance(Y_train)
        
        # 2. Model Training
        print("\n[2/3] Training model...")
        X_train, X_val, Y_train, Y_val = get_train_val_split(X_train, Y_train)
        model, history = train_model(X_train, Y_train, X_val, Y_val)
        
        # Plot training history
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
        
        # 3. Evaluation
        print("\n[3/3] Evaluating model...")
        evaluate_model(model, X_val, Y_val)
        analyze_predictions(model, test_data)
        plot_confusion_matrix(model, X_val, Y_val)
        
        print("\n=== Process Completed Successfully ===")
        
    except Exception as e:
        print(f"\n!!! Process failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()