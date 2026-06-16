from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import sys
import os

def load_class_labels(json_path):
    with open(json_path) as f:
        class_indices = json.load(f)
    # Sort by index (value) to get label list in correct order
    return [label for label, index in sorted(class_indices.items(), key=lambda x: x[1])]

def predict(img_path, model_path, labels_path):
    # Load model
    model = load_model(model_path)

    # Load class labels in correct order
    class_labels = load_class_labels(labels_path)

    # Load and preprocess image
    img = image.load_img(img_path, target_size=(300, 300))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    pred_index = np.argmax(pred)
    pred_label = class_labels[pred_index]
    confidence = round(pred[0][pred_index] * 100, 2)

    return pred_label, confidence

# Example usage
if __name__ == "__main__":
    # Modify these as needed or pass from command line
    img_path = 'test_xray.jpg'
    model_path = 'models/xray_model.h5'
    labels_path = 'models/class_indices.json'

    if not os.path.exists(img_path):
        print(f"Image file '{img_path}' not found.")
        sys.exit(1)

    label, confidence = predict(img_path, model_path, labels_path)
    print(f"Diagnosis: {label} ({confidence}%)")
