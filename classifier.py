import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class ImageCls(nn.Module):
    def __init__(self, classification_cfg: DictConfig, **kwargs):
        super(ImageCls, self).__init__(**kwargs)
        self.feature_proj = nn.Linear(classification_cfg["hidden_dim"], classification_cfg["proj_dim"])
        self.attention_proj = nn.Linear(classification_cfg["hidden_dim"], classification_cfg["proj_dim"])
        self.cls_proj = nn.Linear(classification_cfg["proj_dim"], classification_cfg["num_class"])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.feature_proj(inputs)
        attention = self.attention_proj(inputs)
        attention = F.softmax(attention, dim=-2)
        seq_repr = torch.sum(attention * features, dim=-2)
        logits = self.cls_proj(seq_repr)
        return logits