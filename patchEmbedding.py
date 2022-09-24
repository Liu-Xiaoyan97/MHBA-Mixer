import torch.nn as nn
from omegaconf import DictConfig
from einops.layers.torch import Rearrange


class patchEmbedding(nn.Module):
    def __init__(self, embedding_cfg: DictConfig, **kwargs):
        super(patchEmbedding, self).__init__(**kwargs)
        patch_size = embedding_cfg["patch_size"]
        feature_map = embedding_cfg['feature_map']
        dimension = int(feature_map//patch_size)**2
        in_channels = embedding_cfg["in_channels"]
        hidden_dim = embedding_cfg["hidden_dim"]
        # print(dimension, hidden_dim)
        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(in_channels*patch_size**2, hidden_dim),
            nn.Conv1d(dimension, dimension, 1, stride=1, padding=0),
            nn.GELU(),
            nn.BatchNorm1d(dimension)
        )

    def forward(self, inputs):
        patch_embedding = self.patch_embedding(inputs)
        return patch_embedding