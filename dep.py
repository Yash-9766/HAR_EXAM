from flask import Flask, render_template, request
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load the .h5 model
model_path = "inception_model.h5"
model = tf.keras.models.load_model(model_path)

# List of class names
class_names = [
    'calling', 'clapping', 'cycling', 'dancing', 'drinking', 'eating',
    'fighting', 'hugging', 'laughing', 'listening_to_music', 'running',
    'sitting', 'sleeping', 'texting', 'using_laptop'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None  # To display the selected image
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_pil = Image.open(image_file)
            image_path = os.path.join("static", image_file.filename)  # Save in "static" folder
            image_pil.save(image_path)
            prediction = predict(image_path)
            image_url = "static/" + image_file.filename  # Set the image URL relative to "static" folder
    return render_template("index_har.html", prediction=prediction, image_url=image_url)

def predict(image_path):
    processed_image = load_and_preprocess_input_image(image_path)
    predictions = model.predict(np.expand_dims(processed_image, axis=0))
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_names[predicted_class_index]
    return predicted_class_label

def load_and_preprocess_input_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to match model's input shape
    img_array = img_to_array(img)
    processed_image = preprocess_input(img_array)
    return processed_image

if __name__ == '__main__':
    app.run(debug=True)
