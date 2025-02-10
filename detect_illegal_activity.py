import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r'C:\Users\ASUS\OneDrive\Desktop\AI_Model_Project\ai_model.h5')

# Function to preprocess image
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or unable to read")
        image = cv2.resize(image, (64, 64))
        image = image / 255.0  # Normalize
        return np.expand_dims(image, axis=0)
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

# Function to mark illegal activity in image
def mark_illegal_activity_image(image_path):
    image = cv2.imread(image_path)
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is None:
        return
    
    prediction = model.predict(preprocessed_image)[0][0]
    
    if prediction > 0.5:
        label = 'Crime Activity'
        color = (0, 0, 255)  # Red for illegal activity
    else:
        label = 'Normal'
        color = (0, 255, 0)  # Green for normal
    
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    output_path = image_path.replace(".png", "_labeled.png")
    cv2.imwrite(output_path, image)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    # Path to your new image
    image_path = r'C:\Users\ASUS\OneDrive\Desktop\AI_Model_Project\image4.png'
    
    mark_illegal_activity_image(image_path)