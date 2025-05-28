import torch
import torchvision.transforms as transforms
from PIL import Image

class AgroScanModel:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path):
        try:
            model = torch.load(model_path, map_location=self.device)
            model.eval()
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def predict(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[0][predicted_class].item()

            return {
                'class': predicted_class,
                'confidence': confidence
            }
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")