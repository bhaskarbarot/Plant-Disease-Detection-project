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
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

def load_model(model_path):
    """Load the trained plant disease detection model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedPlantClassifier(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def process_image(image):
    """Process the input image for model inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def get_prediction(model, image_tensor):
    """Get prediction from the model"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    return predicted.item(), probabilities[0]

def get_disease_recommendations():
    """Return the disease recommendations dictionary"""
    return {
        "Apple___Apple_scab": {
            "pesticides": [
                "- Myclobutanil (Organic alternative: Neem oil)",
                "- Captan (Organic alternative: Sulfur spray)",
                "- Propiconazole (Organic alternative: Copper fungicide)"
            ],
            "fertilizers": [
                "- Balanced NPK (10-10-10)",
                "- Calcium-rich fertilizer",
                "- Organic compost tea"
            ],
            "treatments": [
                "1. Remove and destroy infected leaves",
                "2. Improve air circulation by pruning",
                "3. Apply fungicides early in the growing season"
            ]
        },
        "Tomato___Late_blight": {
            "pesticides": [
                "- Chlorothalonil (Organic alternative: Copper spray)",
                "- Mancozeb (Organic alternative: Bacillus subtilis)",
                "- Azoxystrobin (Organic alternative: Potassium bicarbonate)"
            ],
            "fertilizers": [
                "- Low nitrogen, high phosphorus fertilizer",
                "- Seaweed-based fertilizer",
                "- Bone meal"
            ],
            "treatments": [
                "1. Remove infected plants immediately",
                "2. Water at base of plants only",
                "3. Maintain proper plant spacing"
            ]
        },
        "Tomato___Early_blight": {
            "pesticides": [
                "- Copper fungicide (Organic)",
                "- Chlorothalonil (Organic alternative: Neem oil)",
                "- Mancozeb (Organic alternative: Sulfur spray)"
            ],
            "fertilizers": [
                "- Balanced organic fertilizer (5-5-5)",
                "- Calcium nitrate",
                "- Fish emulsion"
            ],
            "treatments": [
                "1. Remove infected lower leaves",
                "2. Mulch around plants",
                "3. Ensure good air circulation"
            ]
        }
        # Add more diseases as needed
    }

def get_treatment_recommendations(disease_name):
    """Get treatment recommendations for a specific disease"""
    disease_recommendations = get_disease_recommendations()
    
    default_recommendations = {
        "pesticides": [
            "- Neem oil (Organic)",
            "- Insecticidal soap (Organic)",
            "- Copper fungicide (Organic)"
        ],
        "fertilizers": [
            "- Balanced organic fertilizer (5-5-5)",
            "- Compost tea",
            "- Worm castings"
        ],
        "treatments": [
            "1. Remove affected plant parts",
            "2. Improve air circulation",
            "3. Maintain proper watering schedule"
        ]
    }
    
    recommendations = disease_recommendations.get(disease_name, default_recommendations)
    
    return f"""
    ### 🌿 Treatment Recommendations for {disease_name.replace('___', ' - ').replace('_', ' ')}

    #### 🧪 Recommended Pesticides:
    {chr(10).join(recommendations['pesticides'])}
    
    #### 💪 Recommended Fertilizers:
    {chr(10).join(recommendations['fertilizers'])}
    
    #### 👨‍🌾 Treatment Methods:
    {chr(10).join(recommendations['treatments'])}
    
    ⚠️ Important Notes:
    - Always follow product labels and local regulations
    - Start with organic options when possible
    - Monitor plant response to treatments
    - Consult with local agricultural extension for specific advice
    """

def main():
    st.set_page_config(page_title="Plant Disease Detection & Treatment", layout="wide")
    
    st.markdown("""
        <style>
        .main { padding: 2rem; }
        .stButton>button { width: 100%; }
        .upload-box {
            border: 2px dashed #4CAF50;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .recommendation-box {
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🌿 Plant Disease Detection & Treatment System")
    st.write("Upload a leaf image to detect diseases and get treatment recommendations")
    
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
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("Analyze Leaf"):
                    with st.spinner("Analyzing image and generating recommendations..."):
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
                            if not is_healthy:
                                st.warning("⚠️ Disease Detected - Generating Treatment Recommendations")
                                recommendations = get_treatment_recommendations(main_prediction)
                                
                                st.markdown("### 🌱 Treatment Recommendations")
                                with st.expander("View Detailed Recommendations", expanded=True):
                                    st.markdown(recommendations)
                            else:
                                st.success("✅ Plant appears to be healthy!")
                                st.markdown("""
                                    ### 🌱 Maintenance Recommendations
                                    - Continue regular watering schedule
                                    - Monitor for any changes in leaf color or texture
                                    - Apply balanced fertilizer as per regular schedule
                                """)
                            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
        ### How to use:
        1. Upload a clear image of a plant leaf
        2. Click "Analyze Leaf" to get:
           - Disease detection results
           - Treatment recommendations
           - Pesticide and fertilizer suggestions
        
        ### Supported Plants:
        Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper,
        Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
    """)

if __name__ == "__main__":
    main()