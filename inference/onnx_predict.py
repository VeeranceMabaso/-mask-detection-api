import onnxruntime as ort
import numpy as np
from PIL import Image

class ONNXMaskDetector:
    def __init__(self, onnx_path):
        """Initialize ONNX mask detector"""
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.img_size = (224, 224)
        
        print(f"ONNX Mask detector loaded successfully!")
    
    def preprocess_image(self, image):
        """Preprocess image for ONNX model"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        image = image.resize(self.img_size)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
        
        return image_array
    
    def predict(self, image):
        """Predict using ONNX model"""
        processed_image = self.preprocess_image(image)
        
        # ONNX inference
        prediction = self.session.run([self.output_name], {self.input_name: processed_image})
        confidence = float(prediction[0][0][0])
        
        if confidence > 0.5:
            class_name = 'without_mask'
            class_confidence = confidence
        else:
            class_name = 'with_mask' 
            class_confidence = 1 - confidence
        
        return {
            'class': class_name,
            'confidence': class_confidence,
            'raw_prediction': confidence,
            'engine': 'onnx'
        }

def test_onnx_detector():
    """Test the ONNX detector"""
    detector = ONNXMaskDetector("../models/mask_detector.onnx")
    
    # Test with a sample image
    import os
    data_path = "../data/raw/with_mask"
    if os.path.exists(data_path):
        sample_image = os.path.join(data_path, os.listdir(data_path)[0])
        result = detector.predict(sample_image)
        print(f"ONNX Prediction: {result['class']}")
        print(f"Confidence: {result['confidence']:.4f}")

if __name__ == "__main__":
    test_onnx_detector()