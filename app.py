from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
from predictions import generate_synthetic_data_logistic
from GrowthGraph import train_lstm_model, prepare_lstm_data
import pickle
import io
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Modified preprocess_image function to accept numpy array instead of path
def preprocess_image(image_array, target_size=(224, 224)):
    # Convert PIL Image to cv2 format
    image = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize image
    resized_image = cv2.resize(gray_image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    normalized_image = resized_image / 255.0
    
    # Convert back to RGB
    rgb_image = cv2.cvtColor((normalized_image * 255).astype('uint8'), cv2.COLOR_GRAY2RGB)
    
    return rgb_image / 255.0  # Normalize to [0, 1]

# Load the trained LSTM model
try:
    lstm_model = tf.keras.models.load_model('lstm_model.h5')
except:
    lstm_model = None

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        # Get the image data from the request
        image_data = request.json['image']
        
        # Convert base64 to image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array and preprocess
        image_np = np.array(image)
        processed_image = preprocess_image(image_np)
        
        try:
            # Generate synthetic data
            synthetic_images, tumor_areas = generate_synthetic_data_logistic(
                processed_image, 
                time_steps=10,
                V0=50,
                V_max=500,
                r=0.1
            )
            
            # Reshape tumor_areas for LSTM input
            X_test = tumor_areas[:-1].reshape(-1, 1, 1)
            
            # Get predictions from LSTM model
            if lstm_model is None:
                raise ValueError("LSTM model not loaded")
            
            # Generate predictions one step at a time
            predicted_areas = []
            current_input = X_test[0].reshape(1, 1, 1)
            
            for _ in range(len(X_test)):
                pred = lstm_model.predict(current_input, verbose=0)
                predicted_areas.append(float(pred[0, 0]))
                current_input = pred.reshape(1, 1, 1)
            
            # Prepare growth data for frontend
            growth_data = {
                'timeSteps': list(range(len(tumor_areas)-1)),
                'trueGrowth': [float(area) for area in tumor_areas[:-1]],
                'predictedGrowth': predicted_areas
            }
            
            print("True growth:", growth_data['trueGrowth'])
            print("Predicted growth:", growth_data['predictedGrowth'])
            
        except Exception as e:
            print(f"Error in growth prediction: {str(e)}")
            raise e
        
        # Convert images to base64 for frontend
        image_predictions = []
        for img in synthetic_images:
            # Convert to grayscale
            gray_img = np.mean(img, axis=2)
            gray_img = (gray_img * 255).astype(np.uint8)
            gray_rgb = np.stack([gray_img] * 3, axis=2)
            
            img_pil = Image.fromarray(gray_rgb)
            buffered = io.BytesIO()
            img_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_predictions.append(img_str)
        
        return jsonify({
            'success': True,
            'predictions': image_predictions,
            'growthData': growth_data
        })
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 