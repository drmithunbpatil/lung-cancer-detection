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

# Label dictionary for clarity
label_dict = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}

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
        img = img.resize((224, 224))
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img

        cnn_pred, xgb_pred, svm_pred, rf_pred, combined_pred = classify_image(file_path)
        result_text = (
            f"CNN Prediction: {cnn_pred}\n"
            f"XGBoost Prediction: {xgb_pred}\n"
            f"SVM Prediction: {svm_pred}\n"
            f"Random Forest Prediction: {rf_pred}\n"
            f"Proposed Novel Ensemble-Based Approach: {combined_pred}"
        )
        result_label.config(text=result_text)

# Create the GUI window
root = tk.Tk()
root.title("Lung Cancer Detection : Proposed  by  Dr Mithun B Patil")
root.geometry("600x700")

# Apply some styling
style = ttk.Style(root)
style.configure('TButton', font=('Helvetica', 12), padding=10)
style.configure('TLabel', font=('Helvetica', 12))

# Add a button to open the file dialog
button = ttk.Button(root, text="Open Image", command=open_file_dialog)
button.pack(pady=20)

# Add a label to display the image
panel = ttk.Label(root)
panel.pack(pady=10)

# Add a label to display the classification results
result_label = ttk.Label(root, text="", wraplength=500, justify='left')
result_label.pack(pady=20)

# Start the GUI event loop
root.mainloop()
