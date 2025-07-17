import os
import sys
import numpy as np
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import (ReduceLROnPlateau, EarlyStopping, ModelCheckpoint) # type: ignore

from sklearn.utils.class_weight import compute_class_weight

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from model.build_model import build_cnn_model
from data.preprocess import load_and_preprocess_data, get_train_val_split

def calculate_class_weights(Y_train):
    """Calculate class weights for imbalanced data"""
    y_labels = np.argmax(Y_train, axis=1)
    class_weights = compute_class_weight('balanced', classes=np.arange(10), y=y_labels)
    return dict(enumerate(class_weights))

def train_model(X_train, Y_train, X_val, Y_val):
    """Train the enhanced CNN model"""
    model = build_cnn_model()
    
    # Enhanced optimizer
    optimizer = Adam(
        learning_rate=Config.LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Enhanced callbacks
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_accuracy',
            patience=2,
            factor=0.5,
            min_lr=0.00001,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            Config.MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest'
    )
    
    # Calculate class weights
    class_weights = calculate_class_weights(Y_train)
    
    print("\nClass weights:", class_weights)
    
    history = model.fit(
        datagen.flow(
            X_train,
            Y_train,
            batch_size=Config.BATCH_SIZE,
            shuffle=True
        ),
        epochs=Config.EPOCHS,
        validation_data=(X_val, Y_val),
        verbose=2,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    return model, history

if __name__ == '__main__':
    print("Starting training...")
    try:
        X_train, Y_train, _ = load_and_preprocess_data()
        X_train, X_val, Y_train, Y_val = get_train_val_split(X_train, Y_train)
        model, history = train_model(X_train, Y_train, X_val, Y_val)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {str(e)}")