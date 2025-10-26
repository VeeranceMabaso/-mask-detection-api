import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

def create_mask_detector_model(img_size=(224, 224), num_classes=1):
    """
    Create a face mask detector using MobileNetV2 as base
    
    Args:
        img_size: Input image size (height, width)
        num_classes: 1 for binary classification (sigmoid output)
    """
    
    # Load pre-trained MobileNetV2 without the top classification layer
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=img_size + (3,)
    )
    
    # Freeze the base model initially (we'll unfreeze later if needed)
    base_model.trainable = False
    
    # Create custom classifier on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),  # Regularization
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),  # More regularization
        layers.Dense(num_classes, activation='sigmoid')  # Sigmoid for binary classification
    ])
    
    # Compile the model - FIXED: Use metric functions instead of strings
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    print("Model created successfully!")
    print(f"Input shape: {img_size + (3,)}")
    print(f"Number of layers: {len(model.layers)}")
    print(f"Total parameters: {model.count_params():,}")
    
    return model

def unfreeze_layers(model, unfreeze_after=100):
    """
    Unfreeze some layers of the base model for fine-tuning
    
    Args:
        model: The compiled model
        unfreeze_after: Unfreeze layers after this index
    """
    base_model = model.layers[0]
    
    # Unfreeze top layers of base model
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    for layer in base_model.layers[:unfreeze_after]:
        layer.trainable = False
    
    # Recompile with lower learning rate - FIXED: Use metric functions
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Lower LR for fine-tuning
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    print("Model unfrozen for fine-tuning!")
    print(f"Trainable layers: {sum([layer.trainable for layer in model.layers[0].layers])}")
    
    return model

if __name__ == "__main__":
    # Test model creation
    model = create_mask_detector_model()
    model.summary()