import requests
import json
import os
import time

class MaskDetectionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def test_health(self):
        """Test health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health")
            print("Health Check:")
            print(f"Status: {response.json()['status']}")
            print(f"Model Loaded: {response.json()['model_loaded']}")
            return True
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def test_detection(self, image_path):
        """Test mask detection on a single image"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                start_time = time.time()
                response = requests.post(f"{self.base_url}/detect", files=files)
                end_time = time.time()
                
            if response.status_code == 200:
                result = response.json()
                print(f"Detection Test - {os.path.basename(image_path)}:")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print(f"Has Mask: {result['has_mask']}")
                print(f"Response Time: {(end_time - start_time)*1000:.2f}ms")
                return True
            else:
                print(f"Detection failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error testing detection: {e}")
            return False
    
    def test_batch_detection(self, image_paths):
        """Test batch detection on multiple images"""
        try:
            files = []
            for path in image_paths:
                files.append(('files', (os.path.basename(path), open(path, 'rb'), 'image/jpeg')))
            
            start_time = time.time()
            response = requests.post(f"{self.base_url}/detect-batch", files=files)
            end_time = time.time()
            
            # Close all files
            for _, file_tuple in files:
                file_tuple[1].close()
            
            if response.status_code == 200:
                result = response.json()
                print(f"Batch Detection Test:")
                print(f"Total Images: {len(image_paths)}")
                print(f"Response Time: {(end_time - start_time)*1000:.2f}ms")
                
                for i, res in enumerate(result['results']):
                    status = "Successful" if res['success'] else "Unsuccessful"
                    if res['success']:
                        print(f"   {status} {os.path.basename(image_paths[i])}: {res['prediction']} ({res['confidence']:.4f})")
                    else:
                        print(f"   {status} {os.path.basename(image_paths[i])}: {res.get('error', 'Unknown error')}")
                
                return True
            else:
                print(f"Batch detection failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error testing batch detection: {e}")
            return False

def main():
    client = MaskDetectionClient()
    
    print("Starting Comprehensive API Tests")
    print("=" * 50)
    
    # Test 1: Health check
    if not client.test_health():
        print("API server might not be running. Please start the server first.")
        return
    
    print("\n" + "=" * 50)
    
    # Test 2: Find test images
    data_path = "../data/raw"
    test_images = []
    
    for class_name in ['with_mask', 'without_mask']:
        class_path = os.path.join(data_path, class_name)
        if os.path.exists(class_path):
            images = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                     if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                # Take up to 2 images from each class
                test_images.extend(images[:2])
    
    if not test_images:
        print("No test images found. Please check your dataset path.")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Test 3: Single detection tests
    print("\n Single Image Detection Tests:")
    print("-" * 30)
    
    for image_path in test_images:
        client.test_detection(image_path)
    
    # Test 4: Batch detection test
    print("\n Batch Detection Test:")
    print("-" * 30)
    client.test_batch_detection(test_images)
    
    print("\n" + "=" * 50)
    print(" All tests completed!")

if __name__ == "__main__":
    main()