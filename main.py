import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Define the model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Preprocess a single image
def preprocess_image(image_path):   
    image = load_img(image_path, target_size=(64, 64))
    image = img_to_array(image)
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)

# Train the model with multiple images
def train_model(model, images, labels, epochs=25):
    model.fit(images, labels, epochs=epochs)
    model.save(r'C:\Users\ASUS\OneDrive\Desktop\AI_Model_Project\ai_model.h5')

if __name__ == "__main__":
    # Paths to your images
    image1_path = r'C:\Users\ASUS\OneDrive\Desktop\AI_Model_Project\image1.png'
    image2_path = r'C:\Users\ASUS\OneDrive\Desktop\AI_Model_Project\image2.png'
    image3_path = r'C:\Users\ASUS\OneDrive\Desktop\AI_Model_Project\image3.jpg'
    image4_path = r'C:\Users\ASUS\OneDrive\Desktop\AI_Model_Project\image4.png'
    # Preprocess the images
    image1 = preprocess_image(image1_path)
    image2 = preprocess_image(image2_path)
    image3 = preprocess_image(image3_path)
    image4 = preprocess_image(image4_path)
    # Combine images and labels
    images = np.vstack([image1, image2, image3, image4])
    labels = np.array([1, 1, 0, 1])  # Assuming image1 is illegal activity (1) and image2 is normal (0)
    
    # Create and train the model
    model = create_model()
    train_model(model, images, labels)