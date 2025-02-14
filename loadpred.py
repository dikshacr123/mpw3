from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load the trained model
with open('predictions.pkl', 'rb') as file:
    model = pickle.load(file)

def preprocess_image(image_data, target_size=(224, 224)):
    """
    Preprocess the uploaded image for model input.
    """
    image = Image.open(BytesIO(image_data))
    image = image.convert('L')  # Convert to grayscale
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    rgb_image = cv2.cvtColor((image_array * 255).astype('uint8'), cv2.COLOR_GRAY2RGB)
    return rgb_image / 255.0

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from the request
        file = request.files['file']
        image_data = file.read()

        # Preprocess the image
        processed_image = preprocess_image(image_data)
        processed_image = processed_image.reshape(1, 224, 224, 3)  # Adjust to model input shape

        # Make prediction
        prediction = model.predict(processed_image)

        # Convert prediction to a format that can be sent as JSON
        prediction_image = (prediction[0] * 255).astype('uint8')
        _, buffer = cv2.imencode('.jpg', prediction_image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'prediction': encoded_image})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
