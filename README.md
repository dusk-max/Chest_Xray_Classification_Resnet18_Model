# Chest X-ray Classification Project

## Overview
This project classifies chest X-ray images into four categories:
- **Normal**
- **Adenocarcinoma**
- **Large Cell Carcinoma**
- **Squamous Cell Carcinoma**

The model is trained using a custom implementation of ResNet-18 and deployed as an interactive web app using Gradio. The app is hosted on Hugging Face Spaces for public access.

---

## Features
- Deep learning-based chest X-ray classification.
- Data augmentation techniques for robust model training.
- Interactive Gradio web app for predictions.
- Deployed app hosted on Hugging Face Spaces.
- Efficient training pipeline with validation and test evaluation.

---

## Directory Structure

```ChestScanProject/
├── Data/
│   ├── train/           # Training images
│   ├── valid/           # Validation images
│   ├── test/            # Testing images
├── model_code.py        # Training script for ResNet-18
├── app.py               # Gradio app for deployment
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation

```

## Dataset

The dataset consists of chest X-ray images categorized into the following classes:

1. **Normal**
2. **Adenocarcinoma**
3. **Large Cell Carcinoma**
4. **Squamous Cell Carcinoma**

### Structure
- The dataset is organized into the following folders:
  - `train/`: Training images for the model.
  - `valid/`: Validation images for model tuning.
  - `test/`: Testing images for evaluating model performance.

### Preprocessing
- All images are resized to **224x224 pixels**.
- Normalization is applied to standardize pixel values.

## Training

The training script is implemented in **model_code.py** using **PyTorch**. 

### Key Features
- **Model Architecture**: Modified ResNet-18 with custom layers for 4-class classification.
- **Loss Function**: CrossEntropyLoss with label smoothing.
- **Optimizer**: SGD with momentum.
- **Scheduler**: ReduceLROnPlateau to adjust the learning rate based on validation loss.

### Steps to Train
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
2. **Place your dataset in the `data\` directory**.
  
3. Run the following script:
 ```bash
   python model_code.py
```
## Deployment

The **Gradio app** (`app.py`) allows users to upload chest X-ray images for classification.

### How to Use the App
1.**Upload an Image**: Drag and drop or upload a chest X-ray image.
2.**View Results**: The app returns the predicted class along with the confidence percentage.

### Run Locally
1.**Install Gradio**:
```bash
pip install gradio
```
2.**Launch the app**:
   ```bash
   python app.py
   ```
3.**Open the link provided by Gradio in your browser**.

### Deployed App

The app is hosted on **Hugging Face Spaces**. You can access it [here](https://huggingface.co/spaces/dusk10/Chest-xray).


## Model Details

- **Base Model**: ResNet-18 pre-trained on ImageNet.

- **Custom Layers**:
  - Fully connected layers with dropout and ReLU activation.
  - Final output layer with 4 neurons (for 4 classes).

- **Metrics**: Achieved **>90% accuracy** on the test set.

## Requirements

- **Python** 3.9+
- **PyTorch**
- **torchvision**
- **Gradio**

### Install dependencies with:
```bash
pip install -r requirements.txt
```

## Results

- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~91%
- **Test Accuracy**: ~90%

## Sample Predictions

- **Predicted class**: Adenocarcinoma
- **Confidence**: 95.32%

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## Contact

For questions or suggestions, please contact at [adityaworks18@gmail.com](mailto:adityaworks18@gmail.com).
