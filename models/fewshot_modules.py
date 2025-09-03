# utils/fewshot_modules.py
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.scale = feature_dim ** -0.5

    def forward(self, support_feat, query_feat):
        attn_weights = torch.softmax(
            torch.bmm(query_feat, support_feat.transpose(1, 2)) * self.scale, dim=-1)
        fused_features = torch.bmm(attn_weights, support_feat)
        return fused_features

class SubspaceClassifier:
    def __init__(self, n_components=5):
        self.n_components = n_components

    def fit_subspace(self, support_features):
        U, S, Vt = torch.linalg.svd(support_features - support_features.mean(dim=0), full_matrices=False)
        subspace = Vt[:self.n_components]
        return subspace

    def classify(self, query_feature, subspaces):
        distances = [torch.norm(query_feature - query_feature @ s.T @ s) for s in subspaces]
        return torch.argmin(torch.stack(distances), dim=0)

class AdaptiveParameterRegulator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, error_feature):
        scale = self.mlp(error_feature)
        return scale

