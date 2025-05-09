import os
import uuid
import cv2
import base64
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from scipy.ndimage import generic_filter

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    return jsonify({"filename": filename})

@app.route("/add_noise", methods=["POST"])
def add_noise():
    data = request.json
    base64_image = data.get("image")
    noise_type = data.get("noiseType")
    params = data.get("params", {})

    if not base64_image or not noise_type:
        return jsonify({"error": "Missing image or noise type"}), 400

    # Strip the header if present
    if ',' in base64_image:
        base64_image = base64_image.split(',')[1]

    # Decode base64 to bytes
    image_bytes = base64.b64decode(base64_image)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return jsonify({"error": "Failed to decode image"}), 400

    noisy_image = apply_noise(image, noise_type, params)

    output_filename = f"noisy_{uuid.uuid4().hex}.png"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    cv2.imwrite(output_path, noisy_image)

    return jsonify({"filename": output_filename})

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def apply_noise(image, noise_type, params):

    if noise_type == "gaussian":
        mean = float(params.get("mean", 0))
        sigma = float(params.get("sigma", 25))
        print('Mean',mean)
        print('Sigma',sigma)
        
        noisy = image + np.random.normal(mean, sigma, image.shape)
        noisy = np.clip(noisy , 0, 255).astype(np.uint8)
        return noisy

    elif noise_type == "saltpepper":
        # Get the values for salt and pepper probabilities
        num_salt = params.get("saltProb", 0)  # Default to 0.02 if not provided
        num_pepper = params.get("pepperProb", 0)  # Default to 0.02 if not provided
       
        num_salt = int(num_salt * image.size)
        num_pepper = int(num_pepper * image.size)

        noisy = image.copy()
        salt_coords = [np.random.randint(0, i , num_salt) for i in image.shape]
        pepper_coords = [np.random.randint(0, i , num_pepper) for i in image.shape]
        noisy[salt_coords[0], salt_coords[1]] = 255
        noisy[pepper_coords[0], pepper_coords[1]] = 0
        return noisy

    elif noise_type == "rayleigh":
        scale = float(params.get("scale", 1.0))
        noise = np.random.rayleigh(scale, image.shape)
        noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy

    elif noise_type == "uniform":
        low = float(params.get("low", -0.05))
        high = float(params.get("high", 0.05))
        noise = np.random.uniform(low, high, image.shape)
        noisy = np.clip(image + noise , 0, 255).astype(np.uint8)
        return noisy

    elif noise_type == "gamma":
        shape = float(params.get("shape", 2.0))
        scale = float(params.get("scale", 1.0))
        noise = np.random.gamma(shape, scale, image.shape)
        noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy

    elif noise_type == "periodic":
        frequency = float(params.get("freq", 10.0))
        rows, cols = image.shape[:2]
        x = np.arange(cols)
        y = np.arange(rows)
        X, Y = np.meshgrid(x, y)
        sinusoidal_pattern = 50 * np.sin(2 * np.pi * frequency * X)  # Adjust amplitude as needed
        noisy = image + sinusoidal_pattern
        return np.clip(noisy, 0, 255).astype(np.uint8)

    else:
        return image
    

@app.route("/apply_filter", methods=["POST"])
def apply_filter():
    data = request.json
    base64_image = data.get("image")
    filter_type = data.get("filterType")
    params = data.get("params", {})

    if not base64_image or not filter_type:
        return jsonify({"error": "Missing image or filter type"}), 400

    if ',' in base64_image:
        base64_image = base64_image.split(',')[1]

    image_bytes = base64.b64decode(base64_image)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return jsonify({"error": "Failed to decode image"}), 400

    filtered_image = apply_filter_function(image, filter_type, params)

    output_filename = f"filtered_{uuid.uuid4().hex}.png"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    cv2.imwrite(output_path, filtered_image)

    return jsonify({"filename": output_filename})

def mean_filter(img, k=3):
    return cv2.blur(img, (k, k))

def harmonic_mean_filter(img, k=3):
    epsilon = 1e-6
    return k ** 2 / (generic_filter(1.0 / (img + epsilon), np.sum, size=k) + epsilon)

def contra_harmonic_filter(img, k=3, Q=1.5):
    num = generic_filter(img ** (Q + 1), np.sum, size=k)
    denom = generic_filter(img ** Q, np.sum, size=k)
    return np.where(denom == 0, img, num / (denom + 1e-6))

def median_filter(img, k=3):
    return cv2.medianBlur(img, k)

def min_filter(img, k=3):
    return generic_filter(img, np.min, size=k)

def max_filter(img, k=3):
    return generic_filter(img, np.max, size=k)

def adaptive_mean_filter(img, k=3):
    return np.where(img < cv2.blur(img, (k, k)), img, cv2.blur(img, (k, k)))

def adaptive_median_filter(img, k=3):
    return median_filter(img, k)

def apply_filter_function(image, filter_type, params):
    k = int(params.get("kernel", 3))
    k = max(1, k)
    k = k if k % 2 == 1 else k + 1  # ensure odd size

    if filter_type == "mean":
        return mean_filter(image, k).astype(np.uint8)

    elif filter_type == "harmonic":
        return harmonic_mean_filter(image.astype(np.float32), k).clip(0, 255).astype(np.uint8)

    elif filter_type == "contraharmonic":
        Q = float(params.get("q", 1.5))
        return contra_harmonic_filter(image.astype(np.float32), k, Q).clip(0, 255).astype(np.uint8)

    elif filter_type == "median":
        return median_filter(image, k).astype(np.uint8)

    elif filter_type == "min":
        return min_filter(image, k).astype(np.uint8)

    elif filter_type == "max":
        return max_filter(image, k).astype(np.uint8)

    elif filter_type == "adaptiveMean":
        return adaptive_mean_filter(image, k).astype(np.uint8)

    elif filter_type == "adaptiveMedian":
        return adaptive_median_filter(image, k).astype(np.uint8)

    elif filter_type == "box":
        width = int(params.get("width", 3))
        height = int(params.get("height", 3))
        width = width if width % 2 == 1 else width + 1
        height = height if height % 2 == 1 else height + 1
        return cv2.blur(image, (width, height)).astype(np.uint8)

    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")

@app.route("/edge_detect", methods=["POST"])
def edge_detect():
    data = request.get_json()
    image_data = data["image"]
    edge_type = data["edgeType"]
    params = data.get("params", {})

    # Decode image
    image_bytes = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Apply edge detection
    try:
        result = apply_edge_function(img, edge_type, params)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Save result
    filename = f"{uuid.uuid4().hex}.png"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(save_path, result)

    return jsonify({"filename": filename})

def apply_edge_function(img, edge_type, params):
    
    if edge_type == "sobel":
        ksize = int(params.get("ksize", 3))
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        magnitude = cv2.magnitude(grad_x, grad_y)
        return np.uint8(np.clip(magnitude, 0, 255))

    elif edge_type == "roberts":
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        grad_x = cv2.filter2D(img, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(img, cv2.CV_64F, kernel_y)
        magnitude = cv2.magnitude(grad_x, grad_y)
        return np.uint8(np.clip(magnitude, 0, 255))

    elif edge_type == "log":
        sigma = float(params.get("sigma", 1.0))
        ksize = int(params.get("ksize", 3))
        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        log = cv2.Laplacian(blurred, cv2.CV_64F)
        return np.uint8(np.clip(np.abs(log), 0, 255))

    elif edge_type == "dog":
        sigma1 = float(params.get("sigma1", 1.0))
        sigma2 = float(params.get("sigma2", 2.0))
        blur1 = cv2.GaussianBlur(img, (0, 0), sigma1)
        blur2 = cv2.GaussianBlur(img, (0, 0), sigma2)
        dog = blur1 - blur2
        return np.uint8(np.clip(dog + 128, 0, 255))

    elif edge_type == "canny":
        low = int(params.get("lowThreshold", 50))
        high = int(params.get("highThreshold", 150))

        canny = cv2.Canny(img, low, high)
        cv2.imshow('Img',canny)
        cv2.waitKey(0)

        return canny

    else:
        raise ValueError(f"Unsupported edge detection method: {edge_type}")


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)

