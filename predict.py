# filepath: c:\Users\NIMASA\Desktop\BELEMA CODE\Lungnosis\predict.py
import torch
from PIL import Image
import os
from torchvision import transforms
from model import TBClassifier

TARGET_SIZE = (224, 224)

def predict_image(image_path, model_path='tb_classifier.pth'):
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please ensure you have trained the model first.")
    
    # Check if image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found.")
    
    try:
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TBClassifier()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Verify and prepare image
        try:
            image = Image.open(image_path)
            # Verify image is valid
            image.verify()
            # Reopen image after verify
            image = Image.open(image_path).convert('RGB')  # Convert to RGB
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image file: {str(e)}")
        
        # Prepare image
        transform = transforms.Compose([
            transforms.Resize(TARGET_SIZE),
            transforms.CenterCrop(TARGET_SIZE),
            transforms.ToTensor()
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs.data, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)
            
        prediction = "Tuberculosis" if predicted.item() == 1 else "Normal"
        confidence_score = confidence[0][predicted.item()].item()
        
        return prediction, confidence_score
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    try:
        # Example usage
        test_image = "./data/Tuberculosis/Tuberculosis-40.png"
        prediction, confidence = predict_image(test_image)
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2%}")
    except Exception as e:
        print(f"Error: {str(e)}")