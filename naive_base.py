import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Function to extract features from images
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img, (50, 50))  # Resize image to a fixed size
    flattened_img = resized_img.flatten()  # Flatten image array
    return flattened_img

# Specify the paths to your image folders
test_folder = r"dataset\test_set"
train_folder = r"dataset\training_set"

# Load images and extract features
X = []
y = []
for folder in [test_folder, train_folder]:
    for label, category in enumerate(['cats', 'dogs']):
        for filename in os.listdir(os.path.join(folder, category)):
            image_path = os.path.join(folder, category, filename)
            features = extract_features(image_path)
            X.append(features)
            y.append(label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predict on test set
y_pred = nb_classifier.predict(X_test)

# Convert numeric labels to string labels
class_labels = {0: 'cat', 1: 'dog'}
predicted_labels = [class_labels[label] for label in y_pred]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print the predicted labels
print("Predicted labels:", predicted_labels)
