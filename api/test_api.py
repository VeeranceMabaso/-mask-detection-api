import requests
import os
import random

def test_api():
    """Test the FastAPI endpoints"""
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    print(" Testing health endpoint...")
    try:
        health_response = requests.get(f"{base_url}/health")
        print(f" Health check: {health_response.json()}")
    except Exception as e:
        print(f" Health check failed: {e}")
        return
    
    # Test root endpoint
    print("\n Testing root endpoint...")
    root_response = requests.get(base_url)
    print(f" Root: {root_response.json()}")
    
    # Test detection with sample images
    print("\n Testing mask detection...")
    
    # Find sample images
    data_path = "../data/raw"
    test_images = []
    
    for class_name in ['with_mask', 'without_mask']:
        class_path = os.path.join(data_path, class_name)
        if os.path.exists(class_path):
            images = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                     if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                test_images.append(random.choice(images))
    
    for image_path in test_images:
        print(f"\n Testing with: {os.path.basename(image_path)}")
        
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(f"{base_url}/detect", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print(f"Has Mask: {result['has_mask']}")
            else:
                print(f"Error: {response.text}")

if __name__ == "__main__":
    test_api()