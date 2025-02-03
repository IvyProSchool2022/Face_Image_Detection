import os
import gdown  # For downloading the model from Google Drive
import numpy as np
import cv2
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Google Drive file ID for the model (Replace with your actual file ID)
MODEL_FILE_ID = "1nyJriPSHilWTz0MEXkerD_wzpfEGLgmC"  # Replace with your actual Google Drive file ID
MODEL_PATH = "face_mask_detector.keras"

def download_model():
    """Download the model from Google Drive if not found locally."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)
        print("Download complete!")

# Ensure the model is available
download_model()

# Load the model
model = load_model(MODEL_PATH)

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

            # ✅ Fix the prediction logic:
            result = "No Mask Detected ❌" if prediction[0][0] < 0.5 else "Mask Detected ✅"

            return render_template("index.html", filename=filename, result=result)

    return render_template("index.html", filename=None, result=None)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve the uploaded image."""
    return redirect(url_for("static", filename="uploads/" + filename))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Allow Railway to assign the port
    app.run(host="0.0.0.0", port=port)
