# Image Classification Web App

A Flask-based web application that allows users to upload an image and uses a pretrained AI model (MobileNetV2) to classify the image and return the top predicted labels with confidence scores.

## Features

- Simple web interface for image upload
- Image classification using TensorFlow's MobileNetV2 model
- Displays the top 3 predicted classes with confidence scores
- Responsive design using Bootstrap

## Screenshot

[Add a screenshot of the application here after running it]

## Setup and Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the application:
   ```
   python app.py
   ```
5. Open your browser and navigate to `http://127.0.0.1:5000/`

## Project Structure

```
project/
│
├── static/uploads/   # Directory for uploaded images
├── templates/        # HTML templates
│   ├── index.html    # Upload form
├── app.py            # Flask application
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## Dependencies

- Flask
- TensorFlow
- Pillow
- NumPy
  
