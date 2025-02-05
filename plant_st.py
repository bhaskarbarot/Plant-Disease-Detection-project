import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import numpy as np

class AdvancedPlantClassifier(nn.Module):
    def __init__(self, num_classes=38):
        super(AdvancedPlantClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedPlantClassifier(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0)
    return image

def get_prediction(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    return predicted.item(), probabilities[0]

def main():
    st.set_page_config(page_title="Plant Disease Detection", layout="wide")
    
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .upload-box {
            border: 2px dashed #4CAF50;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🌿 Plant Disease Detection System")
    st.write("Upload a leaf image to detect plant diseases")
    
    try:
        model = load_model("D:/bhachu/plant_project/plant_disease_model_acc_0.9352.pth")
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated parameter here
                
                if st.button("Detect Disease"):
                    with st.spinner("Analyzing image..."):
                        image_tensor = process_image(image)
                        prediction_idx, probabilities = get_prediction(model, image_tensor)
                        top3_prob, top3_indices = torch.topk(probabilities, 3)
                        
                        with col2:
                            st.subheader("Detection Results")
                            
                            main_prediction = CLASS_NAMES[prediction_idx]
                            plant_name = main_prediction.split('___')[0]
                            condition = main_prediction.split('___')[1].replace('_', ' ')
                            
                            st.markdown(f"""
                                ### Main Prediction
                                - **Plant**: {plant_name}
                                - **Condition**: {condition}
                                - **Confidence**: {probabilities[prediction_idx]*100:.2f}%
                            """)
                            
                            st.markdown("### Top 3 Predictions")
                            for prob, idx in zip(top3_prob, top3_indices):
                                class_name = CLASS_NAMES[idx]
                                plant = class_name.split('___')[0]
                                condition = class_name.split('___')[1].replace('_', ' ')
                                st.progress(float(prob))
                                st.write(f"{plant} - {condition}: {prob*100:.2f}%")
                            
                            is_healthy = "healthy" in CLASS_NAMES[prediction_idx].lower()
                            if is_healthy:
                                st.success("✅ Plant appears to be healthy!")
                            else:
                                st.warning("⚠️ Plant may have a disease")
                                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
        ### How to use:
        1. Upload a clear image of a plant leaf
        2. Click the "Detect Disease" button
        3. View the results and predictions
        
        ### Supported Plants:
        - Apple
        - Blueberry
        - Cherry
        - Corn (Maize)
        - Grape
        - Orange
        - Peach
        - Pepper
        - Potato
        - Raspberry
        - Soybean
        - Squash
        - Strawberry
        - Tomato
    """)

if __name__ == "__main__":
    main()