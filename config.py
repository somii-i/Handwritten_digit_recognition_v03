# Configuration parameters for the project
import os

class Config:
    # Data configuration
    DATA_PATH = 'data'
    TRAIN_FILE = 'train.csv'
    TEST_FILE = 'test.csv'
    
    # Model configuration
    IMAGE_SIZE = 28
    NUM_CLASSES = 10
    BATCH_SIZE = 86
    EPOCHS = 20
    LEARNING_RATE = 0.001
    
    # Training configuration
    VALIDATION_SPLIT = 0.1
    RANDOM_STATE = 42
    
    # Paths
    MODEL_SAVE_PATH = 'model/saved_models/digit_recognizer.h5'
    
    @staticmethod
    def create_dirs():
        os.makedirs('data/', exist_ok=True)
        os.makedirs('model/saved_models/', exist_ok=True)
        os.makedirs('utils/', exist_ok=True)