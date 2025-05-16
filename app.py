from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Create necessary directories if they don't exist
os.makedirs('static/uploads', exist_ok=True)

# Load the pretrained model
model = MobileNetV2(weights='imagenet')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    # Save the uploaded image
    img_path = os.path.join('static/uploads', file.filename)
    file.save(img_path)

    try:
        # Preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        preds = model.predict(img_array)
        decoded_preds = decode_predictions(preds, top=3)
        
        # Get top predictions
        predictions = []
        for i, (imagenet_id, label, score) in enumerate(decoded_preds[0]):
            predictions.append({
                'rank': i + 1,
                'label': label,
                'confidence': f"{score * 100:.2f}%"
            })
        
        return render_template('index.html', predictions=predictions, image_file=file.filename)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)