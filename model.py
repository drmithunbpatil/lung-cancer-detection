import os
import sys
import locale
import numpy as np
import pickle
import warnings

from skimage.feature import hog
from skimage import io, transform
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Set TensorFlow logging level to suppress informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set the default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# Function to load and preprocess images from folders
def load_images_from_folder(folder, max_images=None):
    images = []
    labels = []
    for label, label_folder in enumerate(['Benign cases', 'Malignant cases', 'Normal cases']):
        path = os.path.join(folder, label_folder)
        for i, filename in enumerate(os.listdir(path)):
            if max_images and i >= max_images:
                break
            img_path = os.path.join(path, filename)
            img = io.imread(img_path, as_gray=True)  # Read images as grayscale
            if img is not None:
                img = transform.resize(img, (224, 224))  # Resize images to match VGG16 input size
                images.append(img)
                labels.append(label)  # Assigning labels based on folder names
    return images, labels

# Load images and labels
data_folder = r'D:\PostDoc\lung cancer\Data\The IQ-OTHNCCD lung cancer dataset'
max_images_per_class = 500  # Limit the number of images per class to manage memory usage

images, labels = load_images_from_folder(data_folder, max_images=max_images_per_class)
images = np.array(images, dtype=np.float32).reshape(-1, 224, 224, 1)  # Reshape for CNN input
images = np.repeat(images, 3, axis=-1)  # Convert grayscale to RGB by repeating channels
labels = np.array(labels, dtype=np.int32)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# Reduce batch size to reduce memory usage
batch_size = 16

# Transfer Learning with VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the CNN model
history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), validation_data=(X_test, y_test), epochs=10, verbose=1)

# Save the CNN model
model.save('cnn_model.h5')

# Feature Extraction using HOG
def extract_hog_features(images):
    hog_features = []
    for image in images:
        hog_image = transform.resize(image[..., 0], (100, 100))  # Convert to grayscale and resize
        features = hog(hog_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
        hog_features.append(features)
    return np.array(hog_features)

hog_features_train = extract_hog_features(X_train)
hog_features_test = extract_hog_features(X_test)

# Train and save the XGBoost Classifier
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_classifier.fit(hog_features_train, y_train)
pickle.dump(xgb_classifier, open('xgb_classifier.pkl', 'wb'))

# Train and save the SVM Classifier
from sklearn.svm import SVC
svm_classifier = SVC(kernel='linear', probability=True)
svm_classifier.fit(hog_features_train, y_train)
pickle.dump(svm_classifier, open('svm_classifier.pkl', 'wb'))

# Train and save the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(hog_features_train, y_train)
pickle.dump(rf_classifier, open('rf_classifier.pkl', 'wb'))
