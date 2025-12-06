import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from PIL import Image
from sklearn.decomposition import PCA

EMOTIONS = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
            4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

st.title("Demo: PCA-Guided CNN Nhận diện cảm xúc")
st.write("Đề tài: Ứng dụng PCA giảm nhiễu hỗ trợ CNN")

# Load Models
@st.cache_resource
def load_models():
    try:
        model = tf.keras.models.load_model('model_pca.h5')
        with open('pca_model.pkl', 'rb') as f:
            pca = pickle.load(f)
        return model, pca
    except:
        st.error("Chưa tìm thấy file model!")
        return None, None

model, pca = load_models()

# Upload ảnh
uploaded_file = st.file_uploader("Chọn một ảnh khuôn mặt...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # Xử lý ảnh đầu vào
    image = Image.open(uploaded_file).convert('L')
    image_resized = image.resize((48, 48))
    img_array = np.array(image_resized)
    
    # Chuẩn bị input cho PCA
    img_input = img_array.astype('float32') / 255.0
    img_input = np.expand_dims(img_input, axis=0) # (1, 48, 48)
    img_flat = img_input.reshape(1, -1) # (1, 2304)

    # Qua PCA (Tái tạo)
    img_pca_flat = pca.transform(img_flat)
    img_recon_flat = pca.inverse_transform(img_pca_flat)
    img_recon = img_recon_flat.reshape(48, 48)

    # Hiển thị so sánh
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_resized, caption='Ảnh gốc (Resize 48x48)', width=150)
    with col2:
        st.image(img_recon, caption='Ảnh sau khi qua PCA (Input CNN)', width=150, clamp=True)

    # Dự đoán bằng CNN
    img_final = img_recon.reshape(1, 48, 48, 1)
    prediction = model.predict(img_final)
    max_index = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    # Kết quả
    st.success(f"Dự đoán: **{EMOTIONS[max_index]}** ({confidence*100:.1f}%)")
    
    # Biểu đồ xác suất
    st.bar_chart(dict(zip(EMOTIONS.values(), prediction[0])))