import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

class MaskDetector:
    def __init__(self, model_path):
        """Initialize the mask detector with trained model"""
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = (224, 224)
        self.class_names = ['with_mask', 'without_mask']
        
        print(f"Mask detector loaded successfully!")
        print(f"Input size: {self.img_size}")
    
    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
       
        if isinstance(image, str):
            # Load from file path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(image)
            else:
                image = Image.fromarray(image).convert('RGB')
        
        # Resize and normalize
        image = image.resize(self.img_size)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def predict(self, image):
        """Predict mask presence in image"""
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Predict
        prediction = self.model.predict(processed_image, verbose=0)
        confidence = float(prediction[0][0])
        
        # Determine class
        if confidence > 0.5:
            class_name = 'without_mask'
            class_confidence = confidence
        else:
            class_name = 'with_mask' 
            class_confidence = 1 - confidence
        
        return {
            'class': class_name,
            'confidence': class_confidence,
            'raw_prediction': confidence
        }
    
    def predict_batch(self, image_paths):
        """Predict mask presence for multiple images"""
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['file'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({'file': image_path, 'error': str(e)})
        
        return results

def test_inference():
    """Test the inference on sample images"""
    detector = MaskDetector("../models/mask_detector.h5")
    
    # Test with some sample images from dataset
    import os
    import random
    
    data_path = "../data/raw"
    test_images = []
    
    # Pick 2 random images from each class
    for class_name in ['with_mask', 'without_mask']:
        class_path = os.path.join(data_path, class_name)
        if os.path.exists(class_path):
            images = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                     if f.endswith(('.png', '.jpg', '.jpeg'))]
            test_images.extend(random.sample(images, 2))
    
    print("Testing inference on sample images...")
    results = detector.predict_batch(test_images)
    
    for result in results:
        if 'error' not in result:
            print(f"{os.path.basename(result['file'])}")
            print(f"Prediction: {result['class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print()

if __name__ == "__main__":
    test_inference()