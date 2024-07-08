import os
from flask import Flask, request, jsonify
import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import time

app = Flask(__name__)

# model = load_model('G:\\FYP\\Python Files\\Food_app_files\\models\\Lime.h5')
 model = load_model('models/Lime.h5')
 
desired_categories = ["Citrus canker", "Huanglongbing", "Good"]
fruit_name = "Lime"
IMAGE_SIZE = (256, 256)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/predictdisease', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    allowed_extensions = {'jpg', 'jpeg', 'png'}
    if file.filename.split('.')[-1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file format'}), 400

    file_path = 'temp_' + str(int(time.time())) + '.jpg'
    file.save(file_path)

    try:
        img = preprocess_image(file_path)

        predictions = model.predict(img)
        predicted_class_index = np.argmax(predictions)
        predicted_disease = desired_categories[predicted_class_index]
        confidence = round(100 * np.max(predictions[0]), 2)

        result = {
            'fruit Name': fruit_name,
            'disease': "None" if predicted_disease == "Good" else predicted_disease,
            'quality': "Good" if predicted_disease == "Good" else "Bad",
            'confidence': confidence
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(file_path)

if __name__ == "__main__":
    app.run(debug=True)
