import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r'C:\Users\ASUS\OneDrive\Desktop\AI_Model_Project\ai_model.h5')

# Function to preprocess image
def preprocess_image(image):
    image = cv2.resize(image, (64, 64))
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)

# Function to mark illegal activity in image
def mark_illegal_activity_image(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)[0][0]
    
    if prediction > 0.5:
        label = 'Crime Activity'
        color = (0, 0, 255)  # Red for illegal activity
    else:
        label = 'Normal'
        color = (0, 255, 0)  # Green for normal
    
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image

# Function to process video and mark illegal activity in frames
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        marked_frame = mark_illegal_activity_image(frame)
        
        cv2.imshow('Video', marked_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to your video
    video_path = r'C:\Users\ASUS\OneDrive\Desktop\AI_Model_Project\videoplayback.mp4'
      
    # Process video and mark illegal activity in frames
    process_video(video_path)