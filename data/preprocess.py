import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical # type: ignore
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def load_and_preprocess_data():
    """Load and preprocess data with error handling"""
    try:
        print("Loading data...")
        train = pd.read_csv(os.path.join(Config.DATA_PATH, Config.TRAIN_FILE))
        test = pd.read_csv(os.path.join(Config.DATA_PATH, Config.TEST_FILE))
        
        Y_train = train["label"]
        X_train = train.drop(labels=["label"], axis=1)
        
        X_train = X_train.values.reshape(-1, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1) / 255.0
        test = test.values.reshape(-1, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1) / 255.0
        
        Y_train = to_categorical(Y_train, num_classes=Config.NUM_CLASSES)
        
        print(f"Data loaded - X_train: {X_train.shape}, Y_train: {Y_train.shape}")
        return X_train, Y_train, test
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise


def get_train_val_split(X_train, Y_train):
    from sklearn.model_selection import train_test_split
    return train_test_split(
        X_train, Y_train, 
        test_size=Config.VALIDATION_SPLIT, 
        random_state=Config.RANDOM_STATE
    )

if __name__ == '__main__':
    try:
        X_train, Y_train, test_data = load_and_preprocess_data()
        print("Data preprocessing completed successfully")
    except:
        print("Data preprocessing failed")