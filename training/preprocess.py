import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def create_data_generators(data_path, img_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Create data generators with augmentation for training and validation
    """
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,        # Reduced for more realistic augmentation
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        validation_split=validation_split
    )
    
    # Only rescaling for validation/test data
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=42  # For reproducibility
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        data_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    print("Data generators created successfully!")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Classes: {list(train_generator.class_indices.keys())}")
    
    return train_generator, val_generator

if __name__ == "__main__":
    # Test the data generators
    train_gen, val_gen = create_data_generators("../data/raw")