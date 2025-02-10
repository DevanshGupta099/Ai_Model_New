from flask import Flask, request, send_file
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import io

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

@app.route('/')
def index():
    return open('index.html').read()

@app.route('/process_file', methods=['POST'])
def process_file_route():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(file_path)
            processed_image = mark_illegal_activity_image(image)
            if processed_image is None:
                return 'Error processing image', 500
            
            # Convert processed image to bytes
            _, buffer = cv2.imencode('.png', processed_image)
            io_buf = io.BytesIO(buffer)
            return send_file(io_buf, mimetype='image/png')

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)