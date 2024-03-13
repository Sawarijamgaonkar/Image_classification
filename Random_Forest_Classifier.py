import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load pre-trained VGG16 model without the top layer (fully connected layers)
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Function to extract features using VGG16 model
def extract_vgg_features(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    features = vgg_model.predict(img_array)
    features = features.flatten()
    return features

# Function to load images from a directory and extract VGG features
def load_images_from_dir(directory):
    features = []
    labels = []
    for label, class_name in enumerate(os.listdir(directory)):
        class_dir = os.path.join(directory, class_name)
        for image_filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_filename)
            if image_filename.endswith(('.jpg', '.jpeg', '.png')):
                features.append(extract_vgg_features(image_path))
                labels.append(label)
    return np.array(features), np.array(labels)

# Load training and testing data
train_dir = r"dataset\training_set"
test_dir = r"dataset\test_set"

X_train, y_train = load_images_from_dir(train_dir)
X_test, y_test = load_images_from_dir(test_dir)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=300, max_depth=30, min_samples_split=10, min_samples_leaf=2, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy:", accuracy)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=os.listdir(test_dir), yticklabels=os.listdir(test_dir))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
