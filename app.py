import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import io
from torchvision import transforms
from torchvision.models import resnet18
import time

# Define the model architecture
def initialize_model():
    model = resnet18(weights=None)  # Use 'weights=None' since we're loading custom weights
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(512, 512),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 4)  # 4 classes
    )
    return model

# Load the model once when the app starts
model = initialize_model()
model.load_state_dict(torch.load('Resnet18_model_weights_ChestScan.pth', map_location=torch.device('cpu')))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image here
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class labels
class_names = ['Normal', 'Adenocarcinoma', 'Large Cell Carcinoma', 'Squamous Cell Carcinoma']

# Define the prediction function
def predict(image):
    try:
        # Add a progress simulation
        time.sleep(1)  # Simulate delay for the user experience of progress bar

        # Apply transformation (resize, normalize, etc.)
        img_tensor = transform(image).unsqueeze(0)

        # Make the prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

        # Get prediction label
        prediction = class_names[predicted_class]

        # Return the response
        return {"prediction": prediction, "confidence": round(confidence * 100, 2)}

    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}

# Set up Gradio interface
inputs = gr.Image(type="pil", label="Upload Chest X-ray Image", image_mode="RGB")
outputs = gr.JSON()

# Gradio interface with added features for aesthetics
interface = gr.Interface(fn=predict, inputs=inputs, outputs=outputs, live=True)

# Launch the Gradio interface
interface.launch()