import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import pickle

# Load dữ liệu FER2013
def load_data(file_path='fer2013.csv'):
    print(f"Đang tải dữ liệu từ {file_path}")
    try:
        data = pd.read_csv(file_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            faces.append(face)
            
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1) # (N, 48, 48, 1)
        
        # Chuẩn hóa về [0, 1]
        faces = faces.astype('float32') / 255.0
        emotions = data['emotion'].values
        
        print(f"Tải thành công {len(faces)} ảnh.")
        return faces, emotions
    except FileNotFoundError:
        print("LỖI: Không tìm thấy file fer2013.csv.")
        return None, None

# Class quản lý PCA
class PCAHandler:
    def __init__(self, n_components=0.98):
        self.pca = PCA(n_components=n_components)
        self.input_shape = (48, 48)

    def fit(self, X_train):
        # Flatten dữ liệu để fit vào PCA: (N, 48, 48, 1) -> (N, 2304)
        N, H, W, C = X_train.shape
        X_flat = X_train.reshape(N, -1)
        print("Đang huấn luyện PCA (Fit)...")
        self.pca.fit(X_flat)
        print(f" PCA đã giữ lại {self.pca.n_components_} thành phần chính.")

    def process_image(self, X):
        # Nén -> Tái tạo -> Reshape
        N, H, W, C = X.shape
        X_flat = X.reshape(N, -1)
        
        # Transform (Nén) và Inverse Transform (Tái tạo - Khử nhiễu)
        X_pca = self.pca.transform(X_flat)
        X_recon = self.pca.inverse_transform(X_pca)
        
        # Reshape
        return X_recon.reshape(N, H, W, C)

    def save(self, filename='pca_model.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.pca, f)
        print(f"Đã lưu model PCA vào {filename}")

    def load(self, filename='pca_model.pkl'):
        with open(filename, 'rb') as f:
            self.pca = pickle.load(f)