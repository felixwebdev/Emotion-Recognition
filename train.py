import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import load_data, PCAHandler
from model import build_cnn
import os

# 1. Cấu hình
EPOCHS = 30 
BATCH_SIZE = 64

# 2. Load Dữ liệu
X, y = load_data('fer2013.csv')
if X is None:
    exit()

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RAW CNN
print("\nBẮT ĐẦU HUẤN LUYỆN RAW CNN")
model_raw = build_cnn()
history_raw = model_raw.fit(X_train, y_train, 
                            validation_data=(X_test, y_test),
                            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
model_raw.save('model_raw.h5')

# PCA-GUIDED CNN
print("\nBẮT ĐẦU HUẤN LUYỆN PCA-GUIDED CNN")

# Train PCA
pca_handler = PCAHandler(n_components=0.98)
pca_handler.fit(X_train) 
pca_handler.save('pca_model.pkl') 

# Tái tạo ảnh
print("Đang tái tạo ảnh qua PCA")
X_train_pca = pca_handler.process_image(X_train)
X_test_pca = pca_handler.process_image(X_test)

# Train CNN
model_pca = build_cnn()
history_pca = model_pca.fit(X_train_pca, y_train, 
                            validation_data=(X_test_pca, y_test),
                            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
model_pca.save('model_pca.h5')

def plot_comparison(h_raw, h_pca):
    acc_raw = h_raw.history['val_accuracy']
    acc_pca = h_pca.history['val_accuracy']
    loss_raw = h_raw.history['val_loss']
    loss_pca = h_pca.history['val_loss']
    epochs_range = range(1, len(acc_raw) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc_raw, label='Raw CNN Val Acc')
    plt.plot(epochs_range, acc_pca, label='PCA+CNN Val Acc', linestyle='--')
    plt.title('So sánh Accuracy (Validation)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss_raw, label='Raw CNN Val Loss')
    plt.plot(epochs_range, loss_pca, label='PCA+CNN Val Loss', linestyle='--')
    plt.title('So sánh Loss (Validation)')
    plt.legend()
    
    plt.savefig('ket_qua_so_sanh.png')
    print("Đã lưu biểu đồ mới: ket_qua_so_sanh.png")
    plt.show()

plot_comparison(history_raw, history_pca)