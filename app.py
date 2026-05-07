import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. UI Configuration
st.set_page_config(page_title="Plant Pathology AI", page_icon="🌿", layout="centered")
st.title("🌿 AI Plant Pathology System")
st.write("Upload a leaf image to detect potential diseases instantly.")

# 2. Load Model (Cached for performance)
@st.cache_resource
def load_model():
    # Make sure this matches your downloaded file name
    return tf.keras.models.load_model('plant_doctor.keras')

model = load_model()

CLASS_NAMES = [
 'Apple___healthy',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Potato___Late_blight',
 'Tomato___Target_Spot',
 'Grape___Black_rot',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Tomato___Leaf_Mold',
 'Tomato___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Tomato___healthy',
 'Blueberry___healthy',
 'Peach___Bacterial_spot',
 'Apple___Black_rot',
 'Corn_(maize)___healthy',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Strawberry___Leaf_scorch',
 'Potato___healthy',
 'Apple___Apple_scab',
 'Grape___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Peach___healthy',
 'Potato___Early_blight',
 'Cherry_(including_sour)___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Soybean___healthy',
 'Tomato___Tomato_mosaic_virus',
 'Strawberry___healthy',
 'Apple___Cedar_apple_rust',
 'Raspberry___healthy',
 'Tomato___Late_blight',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Grape___Esca_(Black_Measles)',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Squash___Powdery_mildew',
 'Tomato___Septoria_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Tomato___Early_blight'
]

# 3. Inference Engine
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Scanned Image', use_column_width=True)
    
    with st.spinner('Analyzing patterns...'):
        # Preprocessing (Math normalization)
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array / 255.0
        
        # Prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        confidence = round(100 * (np.max(predictions[0])), 2)
        predicted_class = CLASS_NAMES[predicted_index]
        
        # 4. Smart Routing (Confidence Threshold)
        st.divider()
        if confidence < 75.0:
            st.warning(f"⚠️ **Low Confidence ({confidence}%)**")
            st.write("The AI is uncertain. Ensure the image is clear, well-lit, and actually contains a leaf from the supported database.")
        else:
            st.success(f"**Diagnosis:** {predicted_class.replace('___', ' - ').replace('_', ' ')}")
            st.info(f"**Confidence:** {confidence}%")