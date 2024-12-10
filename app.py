import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)

# Set the folder for saving uploaded files
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = load_model('HandDigitRecorgnitionModel.h5')

def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize image to 28x28 pixels
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    
    # Flatten the image to match (None, 784)
    img_array = img_array.reshape(1, 28*28)  # Flatten to shape (1, 784)
    
    return img_array


# Route for home page and image upload
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'digit_image' not in request.files:
            return redirect(request.url)
        file = request.files['digit_image']
        if file.filename == '':
            return redirect(request.url)
        
        # Save the uploaded file
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image and make a prediction
            img_array = preprocess_image(file_path)
            predictions = model.predict(img_array)
            predicted_digit = np.argmax(predictions, axis=1)[0]

            # Redirect to result page
            return render_template('result.html', filename=filename, predicted_digit=predicted_digit)
    
    return render_template('upload.html')

# Route for displaying the result
@app.route('/result')
def result():
    return render_template('result.html')

# To serve the uploaded image in the result page
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)
