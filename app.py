import os
import numpy as np
import cv2
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model(r"F:\datasets\Face_Mask_Detection\face_mask_detector.keras")

# Define upload folder and allowed extensions
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Load and preprocess image for prediction."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model input
    return img

@app.route("/", methods=["GET", "POST"])
def upload_image():
    """Handle image upload and prediction."""
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        
        if file.filename == "":
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Process the image and make a prediction
            img = preprocess_image(file_path)
            prediction = model.predict(img)

            # Interpret the result
            result = "No Mask Detected ❌" if prediction[0][0] < 0 else "Mask Detected ✅"

            return render_template("index.html", filename=filename, result=result)
    
    return render_template("index.html", filename=None, result=None)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve the uploaded image."""
    return redirect(url_for("static", filename="uploads/" + filename))

if __name__ == "__main__":
    app.run(debug=True)