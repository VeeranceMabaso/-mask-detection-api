import tensorflow as tf
import tf2onnx
import onnxruntime as ort
import numpy as np
import time

def convert_to_onnx(model_path, onnx_path):
    """Convert TensorFlow model to ONNX for faster inference"""
    
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Convert to ONNX
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    
    # Save ONNX model
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    
    print(f"Model converted to ONNX: {onnx_path}")
    return onnx_path

def test_onnx_performance(onnx_path, test_image):
    """Test ONNX model performance"""
    
    # Create ONNX runtime session
    session = ort.InferenceSession(onnx_path)
    
    # Prepare input
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Warm up
    for _ in range(10):
        session.run([output_name], {input_name: test_image})
    
    # Benchmark
    times = []
    for _ in range(100):
        start = time.time()
        session.run([output_name], {input_name: test_image})
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000  # Convert to milliseconds
    print(f"ONNX Average Inference Time: {avg_time:.2f}ms")
    return avg_time

if __name__ == "__main__":
    # Convert model
    onnx_path = convert_to_onnx("../models/mask_detector.h5", "../models/mask_detector.onnx")
    
    # Create a test image
    test_image = np.random.randn(1, 224, 224, 3).astype(np.float32)
    
    # Test performance
    onnx_time = test_onnx_performance(onnx_path, test_image)
    
    # Compare with TensorFlow
    model = tf.keras.models.load_model("../models/mask_detector.h5")
    
    
    for _ in range(10):
        model.predict(test_image, verbose=0)
    
    tf_times = []
    for _ in range(100):
        start = time.time()
        model.predict(test_image, verbose=0)
        tf_times.append(time.time() - start)
    
    tf_avg_time = np.mean(tf_times) * 1000
    print(f"TensorFlow Average Inference Time: {tf_avg_time:.2f}ms")
    print(f"Speedup: {tf_avg_time/onnx_time:.2f}x faster!")