from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
from predictions import generate_synthetic_data_logistic
from GrowthGraph import logistic_growth
import pickle
import io
from PIL import Image

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

# Load the trained model
try:
    model = pickle.load(open('predictions.pkl', 'rb'))
except:
    model = None

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
        
        # Use the same parameters as in your original model
        time_steps = 10  # Changed to match original
        V0 = 50         # Initial tumor size
        V_max = 500     # Maximum tumor size
        r = 0.1         # Growth rate
        
        # Generate synthetic data with the same parameters
        synthetic_images, tumor_areas = generate_synthetic_data_logistic(
            processed_image, 
            time_steps,
            V0=V0,
            V_max=V_max,
            r=r
        )
        
        # Use actual tumor areas for growth data
        growth_data = []
        for t, area in enumerate(tumor_areas):
            growth_data.append({
                'timeStep': t + 1,
                'value': float(area)  # Use actual computed areas
            })
        
        # Convert images to base64 for sending back to frontend
        image_predictions = []
        for img in synthetic_images:
            # Convert to grayscale by taking average of channels
            gray_img = np.mean(img, axis=2)
            
            # Normalize and convert to uint8
            gray_img = (gray_img * 255).astype(np.uint8)
            
            # Convert to RGB (all channels will be the same - creating grayscale)
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