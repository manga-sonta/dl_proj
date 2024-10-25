import streamlit as st
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

# Define your U-Net model class here or import it if defined elsewhere
# For this example, we'll define a simple U-Net model (adjust as needed)
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        # Define the layers of your U-Net model here
        # This is a placeholder model; replace with your actual model architecture
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv_last = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv_last(x)
        return x

# Cache the model loading to avoid reloading on every run
@st.cache_resource
def load_model():
    model = UNet(num_classes=5)  # Adjust num_classes as per your model
    # Load the trained model weights
    model.load_state_dict(torch.load('unet_model_complete.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocessing function
def preprocess_image(image, transform):
    """
    Load and preprocess the image.
    :param image: PIL Image.
    :param transform: Transformation to apply.
    :return: Preprocessed image tensor.
    """
    # Convert image to grayscale
    image = image.convert('L')
    # Apply the transformations
    image = transform(image)
    # Add batch dimension
    image = image.unsqueeze(0)
    return image

# Prediction function
def predict_classes_in_image(unet_model, image_tensor):
    """
    Run the model on the unseen image and return unique predicted classes.
    :param unet_model: Trained U-Net model.
    :param image_tensor: Preprocessed image tensor.
    :return: List of unique class indices predicted.
    """
    with torch.no_grad():
        # Forward pass through the model
        output = unet_model(image_tensor)
        # Get the predicted class for each pixel
        predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        # Get unique classes present in the predicted mask
        unique_classes = np.unique(predicted_mask)
    return unique_classes

# Display function
def display_classes(unique_classes):
    """
    Display the unique predicted classes in human-readable form.
    :param unique_classes: List of unique class indices.
    """
    class_names = {
        0: 'Background',
        1: 'Liver',
        2: 'Right Kidney',
        3: 'Left Kidney',
        4: 'Spleen'
    }
    st.subheader("Classes present in the image based on prediction:")
    for cls in unique_classes:
        st.write(f"- **Class {cls}**: {class_names.get(cls, 'Unknown')}")

def main():
    st.title("Medical Image Segmentation with U-Net")
    st.write("Upload a medical image to see the predicted organ classes.")

    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # Add normalization if needed
        # transforms.Normalize((0.5,), (0.5,)),
    ])

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Open the image file
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        image_tensor = preprocess_image(image, transform)

        # Load the model
        model = load_model()

        # Predict
        unique_classes = predict_classes_in_image(model, image_tensor)

        # Display the results
        display_classes(unique_classes)

if __name__ == '__main__':
    main()
