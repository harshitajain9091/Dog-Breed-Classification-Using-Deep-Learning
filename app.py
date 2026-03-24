from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import json
import os

app = Flask(__name__)

# Load breed information
with open('breed_info.json', 'r') as f:
    breed_info = json.load(f)

# Breed class mapping
BREED_CLASSES = list(breed_info.keys())

# Try to load the model
model_loaded = False
use_tflite = False
interpreter = None
model = None

print("\n" + "="*60)
print("DOG BREED CLASSIFIER - SERVER STARTING")
print("="*60)

try:
    # Try to load TFLite model first
    if os.path.exists('models/dog_breed_model.tflite'):
        print("📱 Loading TFLite model...")
        interpreter = tf.lite.Interpreter(model_path="models/dog_breed_model.tflite")
        interpreter.allocate_tensors()
        model_loaded = True
        use_tflite = True
        print("✅ TFLite model loaded successfully!")
    elif os.path.exists('models/dog_breed_model.h5'):
        # Fallback to Keras model
        print("📦 Loading Keras model...")
        model = tf.keras.models.load_model('models/dog_breed_model.h5')
        model_loaded = True
        use_tflite = False
        print("✅ Keras model loaded successfully!")
    else:
        print("⚠️  No model found! Using demo mode with random predictions.")
        print("   To train a model, run: python train_model.py")
except Exception as e:
    print(f"⚠️  Error loading model: {e}")
    print("   Using demo mode with random predictions.")

print("="*60 + "\n")

def preprocess_image(image_bytes):
    """Preprocess image for model prediction"""
    # Open image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 224x224 (MobileNetV2 input size)
    image = image.resize((224, 224))
    
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array.astype(np.float32)

def predict_with_tflite(image_bytes):
    """Predict using TFLite model"""
    image_array = preprocess_image(image_bytes)
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    return predictions[0]

def predict_with_keras(image_bytes):
    """Predict using Keras model"""
    image_array = preprocess_image(image_bytes)
    predictions = model.predict(image_array, verbose=0)
    return predictions[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return breed prediction"""
    try:
        # Get image from request
        data = request.get_json()
        image_data = data['image'].split(',')[1]  # Remove base64 header
        image_bytes = base64.b64decode(image_data)
        
        # Make prediction
        if not model_loaded:
            # Demo mode - return random predictions
            import random
            predicted_breed = random.choice(BREED_CLASSES)
            confidence = random.uniform(0.6, 0.95)
            print(f"🎲 Demo mode - Random prediction: {predicted_breed} ({confidence:.1%} confidence)")
        else:
            # Use actual model
            if use_tflite:
                predictions = predict_with_tflite(image_bytes)
            else:
                predictions = predict_with_keras(image_bytes)
            
            # Get predicted class
            predicted_index = np.argmax(predictions)
            confidence = float(predictions[predicted_index])
            predicted_breed = BREED_CLASSES[predicted_index]
            print(f"🤖 Model prediction: {predicted_breed} ({confidence:.1%} confidence)")
        
        # Get breed information
        breed_details = breed_info.get(predicted_breed, breed_info[BREED_CLASSES[0]])
        
        # Prepare response
        response = {
            'success': True,
            'breed': breed_details['name'],
            'breed_key': predicted_breed,
            'confidence': f"{confidence * 100:.1f}%",
            'details': {
                'origin': breed_details['origin'],
                'temperament': breed_details['temperament'],
                'life_span': breed_details['life_span'],
                'height': breed_details['height'],
                'weight': breed_details['weight'],
                'description': breed_details['description'],
                'fun_fact': breed_details['fun_fact']
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("📍 Open your browser and go to: http://localhost:5000")
    print("📸 Take a photo or upload an image to classify dog breed")
    print("⏹️  Press Ctrl+C to stop the server\n")
    print("="*60)
    print()
    app.run(debug=True, host='0.0.0.0', port=5000)