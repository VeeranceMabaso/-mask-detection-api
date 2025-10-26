import tensorflow as tf
import mlflow
import mlflow.tensorflow
import os
import datetime
from model import create_mask_detector_model, unfreeze_layers
from preprocess import create_data_generators

def train_model():
    """Train the mask detection model with MLflow tracking"""
    
    # Set up paths
    data_path = "../data/raw"
    model_save_path = "../models/mask_detector.h5"
    os.makedirs("../models", exist_ok=True)
    
    # MLflow setup
    mlflow.set_tracking_uri("../mlruns")  # Local storage
    mlflow.set_experiment("face_mask_detection")
    
    # Training parameters
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS_INITIAL = 10
    EPOCHS_FINE_TUNE = 5  # Reduced for testing
    
    # Create data generators
    train_gen, val_gen = create_data_generators(
        data_path, 
        img_size=IMG_SIZE, 
        batch_size=BATCH_SIZE
    )
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs_initial": EPOCHS_INITIAL,
            "epochs_fine_tune": EPOCHS_FINE_TUNE,
            "base_model": "MobileNetV2",
            "optimizer": "Adam"
        })
        
        # Create and train initial model
        print("Creating initial model...")
        model = create_mask_detector_model(IMG_SIZE)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=3,  # Reduced for testing
                restore_best_weights=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                model_save_path,
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Initial training (frozen base)
        print("Initial training (frozen base)...")
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        
        try:
            history_initial = model.fit(
                train_gen,
                epochs=EPOCHS_INITIAL,
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1,
                steps_per_epoch=len(train_gen),
                validation_steps=len(val_gen)
            )
            
            # Fine-tuning (unfreeze some layers)
            print("Fine-tuning (unfreezing layers)...")
            model = unfreeze_layers(model)
            
            history_fine_tune = model.fit(
                train_gen,
                epochs=EPOCHS_FINE_TUNE,
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1,
                steps_per_epoch=len(train_gen),
                validation_steps=len(val_gen)
            )
            
            # Save final model
            model.save(model_save_path)
            print(f"Model saved to {model_save_path}")
            
            # Log metrics
            final_val_accuracy = history_fine_tune.history['val_accuracy'][-1]
            final_val_loss = history_fine_tune.history['val_loss'][-1]
            
            mlflow.log_metrics({
                "final_val_accuracy": final_val_accuracy,
                "final_val_loss": final_val_loss,
                "best_val_accuracy": max(history_fine_tune.history['val_accuracy'])
            })
            
            # Log model
            mlflow.tensorflow.log_model(model, "model")
            
            # Log artifacts
            mlflow.log_artifact(model_save_path)
            
            print("Training completed!")
            print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")
            print(f"Final Validation Loss: {final_val_loss:.4f}")
            
            return model, history_initial, history_fine_tune
            
        except Exception as e:
            print(f"Training failed: {e}")
            # Save a simple model for testing if training fails
            print("Saving basic model for testing...")
            model.save(model_save_path)
            raise e

if __name__ == "__main__":
    # Start training
    try:
        model, history_initial, history_fine_tune = train_model()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")