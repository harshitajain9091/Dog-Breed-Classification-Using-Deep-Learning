import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import os
import numpy as np

print("="*60)
print("DOG BREED CLASSIFIER - MODEL TRAINING")
print("="*60)
print(f"TensorFlow version: {tf.__version__}")
print()

def create_model(num_classes=6):
    """Create a CNN model using Transfer Learning with MobileNetV2"""
    print("📦 Creating model architecture...")
    
    # Load pre-trained MobileNetV2 model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom classification layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_demo_model():
    """Create a demo model with random weights for testing"""
    print("🎯 Creating demo model...")
    model = create_model(num_classes=6)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create dummy data for demonstration
    print("📊 Creating dummy training data...")
    dummy_images = np.random.rand(100, 224, 224, 3)
    dummy_labels = np.random.randint(0, 6, 100)
    dummy_labels = tf.keras.utils.to_categorical(dummy_labels, 6)
    
    # Train for a few epochs
    print("🏋️ Training demo model (2 epochs with random data)...")
    model.fit(
        dummy_images, 
        dummy_labels,
        epochs=2,
        batch_size=32,
        verbose=1
    )
    
    # Create models folder if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the model
    print("💾 Saving model...")
    model.save('models/dog_breed_model.h5')
    print("✅ Model saved as 'models/dog_breed_model.h5'")
    
    # Convert to TFLite for better performance
    print("🔄 Converting to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open('models/dog_breed_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("✅ TFLite model saved as 'models/dog_breed_model.tflite'")
    print()
    print("⚠️  NOTE: This is a DEMO model trained on random data!")
    print("📌 For real dog breed classification, you need to:")
    print("   1. Download a real dataset from Kaggle (Stanford Dogs Dataset)")
    print("   2. Train on actual dog images for several hours")
    print("   3. Or download a pre-trained model from TensorFlow Hub")
    print()

if __name__ == "__main__":
    create_demo_model()
    print("✅ Training complete! You can now run: python app.py")