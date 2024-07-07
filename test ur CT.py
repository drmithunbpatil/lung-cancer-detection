import os
import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from skimage import io, transform
from skimage.feature import hog
from tensorflow.keras.models import load_model

# Load saved models
cnn_model = load_model('cnn_model.h5')
xgb_classifier = pickle.load(open('xgb_classifier.pkl', 'rb'))
svm_classifier = pickle.load(open('svm_classifier.pkl', 'rb'))
rf_classifier = pickle.load(open('rf_classifier.pkl', 'rb'))

# Updated label dictionary for more clarity
label_dict = {
    0: 'Benign : its alerting stage kindly start the treatment',
    1: 'Malignant :Its at Dangerous stage alert to doctor',
    2: 'Normal : No issue you have healthy lungs'
}

# Function to classify a single image
def classify_image(image_path):
    img = io.imread(image_path, as_gray=True)  # Read the image as grayscale
    img = transform.resize(img, (224, 224))  # Resize to match VGG16 input size
    img = np.expand_dims(img, axis=2)
    img = np.repeat(img, 3, axis=2)  # Convert grayscale to RGB by repeating channels
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # CNN prediction
    cnn_probabilities = cnn_model.predict(img)
    cnn_prediction = np.argmax(cnn_probabilities, axis=1)[0]

    # HOG feature extraction
    hog_image = transform.resize(img[0, ..., 0], (100, 100))
    hog_features = hog(hog_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
    hog_features = np.expand_dims(hog_features, axis=0)

    # XGBoost prediction
    xgb_probabilities = xgb_classifier.predict_proba(hog_features)
    xgb_prediction = np.argmax(xgb_probabilities, axis=1)[0]

    # SVM prediction
    svm_probabilities = svm_classifier.predict_proba(hog_features)
    svm_prediction = np.argmax(svm_probabilities, axis=1)[0]

    # Random Forest prediction
    rf_probabilities = rf_classifier.predict_proba(hog_features)
    rf_prediction = np.argmax(rf_probabilities, axis=1)[0]

    # Combined prediction using majority vote
    combined_prediction = np.bincount([cnn_prediction, xgb_prediction, svm_prediction, rf_prediction]).argmax()

    return label_dict[cnn_prediction], label_dict[xgb_prediction], label_dict[svm_prediction], label_dict[rf_prediction], label_dict[combined_prediction]

# Function to open a file dialog and classify an image
def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img

        cnn_pred, xgb_pred, svm_pred, rf_pred, combined_pred = classify_image(file_path)
        result_text = (
            "Earlier Methods results\n"
            f"CNN Prediction: {cnn_pred}\n"
            f"XGBoost Prediction: {xgb_pred}\n"
            f"SVM Prediction: {svm_pred}\n"
            f"Random Forest Prediction: {rf_pred}\n"
            "------------------------------------------------------------------------------\n"
            "Proposed Method Results\n"
            f"Novel Proposed Approach: {combined_pred}"
        )
        result_label.config(text=result_text)

# Create the GUI window
root = tk.Tk()
root.title("Lung Cancer Detection System Proposed By Dr Mithun B. patil")
root.geometry("800x600")

# Load background image
background_image = Image.open("background.jpg")  # Ensure you have a light background image named 'background.jpg'
background_image = background_image.resize((800, 600), Image.Resampling.LANCZOS)
background_image = ImageTk.PhotoImage(background_image)

background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

# Apply some styling
style = ttk.Style(root)
style.configure('TButton', font=('Helvetica', 14), padding=10)
style.configure('TLabel', font=('Helvetica', 14), background='white')

# Add a title
title_label = ttk.Label(root, text="Lung Cancer Detection", font=('Helvetica', 24, 'bold'), background='#ADD8E6')
title_label.pack(pady=20)

# Add a button to open the file dialog
button = ttk.Button(root, text=" Load you CT Scan Image here", command=open_file_dialog)
button.pack(pady=20)

# Add a label to display the image
panel = ttk.Label(root, background='white')
panel.pack(pady=10)

# Add a label to display the classification results
result_label = ttk.Label(root, text="", wraplength=800, justify='center', background='red')
result_label.pack(pady=20)

# Start the GUI event loop
root.mainloop()
