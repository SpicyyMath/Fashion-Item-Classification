import torch
from torchvision import models
from sklearn.cluster import DBSCAN
from sklearn.svm import SVC
import numpy as np

# 特征提取器
class FeatureExtractor:
    def __init__(self, device='cuda'):
        # 使用预训练的 VGG16，仅保留卷积层,也可以用resnet
        vgg16 = models.vgg16(pretrained=True)
        self.features = torch.nn.Sequential(*list(vgg16.features.children())).to(device)
        self.device = device

    def extract(self, image):
        image = image.to(self.device)
        with torch.no_grad():
            feat = self.features(image.unsqueeze(0)) 
            feat = feat.view(-1)
        return feat.cpu().numpy() 

# 聚类模型
class ClusterModel:
    def __init__(self, eps=1.0, min_samples=5):
        # 初始化 DBSCAN 
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    def fit_predict(self, features):
        # 对特征进行聚类并返回标签
        return self.dbscan.fit_predict(features)

# 分类器
class Classifier:
    # CORRECT __init__ accepting parameters
    def __init__(self, kernel='rbf', C=1.0, probability=True, random_state=42, class_weight=None): # Add more params if needed
        print(f"--- Initializing Classifier from models/cluster_model.py (Kernel: {kernel}, C: {C}) ---") # Add a print statement HERE to be SURE this version is used
        self.model = SVC(kernel=kernel,
                         C=C,
                         probability=probability,
                         random_state=random_state,
                         class_weight=class_weight) # Pass arguments to SVC
        self._fitted_labels = []

    def fit(self, X, y):
        # (Your fit method logic - ensure X is 2D numpy array)
        X_np = np.array(X)
        if X_np.ndim != 2:
            # Handle flattening or reshaping as needed
            num_samples = X_np.shape[0]
            X_np = X_np.reshape(num_samples, -1)
        self.model.fit(X_np, y)
        self._fitted_labels = list(self.model.classes_)
        # print(f"Classifier fit completed. Labels seen: {self._fitted_labels}") # Optional debug

    def predict(self, X):
        # (Your predict method logic - ensure X is 2D numpy array)
        X_np = np.array(X)
        if X_np.ndim != 2:
             # Handle flattening or reshaping as needed
             num_samples = X_np.shape[0]
             X_np = X_np.reshape(num_samples, -1)
        return self.model.predict(X_np)

    # Add predict_proba if needed
    def predict_proba(self, X):
         X_np = np.array(X)
         if X_np.ndim != 2:
              num_samples = X_np.shape[0]
              X_np = X_np.reshape(num_samples, -1)
         return self.model.predict_proba(X_np)

# Other classes like FeatureExtractor might also be in this file...