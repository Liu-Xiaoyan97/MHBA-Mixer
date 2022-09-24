import torch

from Modules import TCAMixer
from omegaconf import DictConfig
import torch.nn as nn

from classifier import ImageCls
from patchEmbedding import patchEmbedding


class TCAMixerImageCls(nn.Module):
    def __init__(self, mode, embedding_cfg: DictConfig, backbone_cfg: DictConfig, classification_cfg: DictConfig, **kwargs):
        super(TCAMixerImageCls, self).__init__(**kwargs)
        # max_seq_len = embedding_cfg['in_channels']*int(embedding_cfg['feature_map'] // embedding_cfg['patch_size']) ** 2
        max_seq_len = int(embedding_cfg['feature_map'] // embedding_cfg['patch_size']) ** 2
        self.patch_embedding = patchEmbedding(embedding_cfg)
        self.mixers = TCAMixer(mode, backbone_cfg, max_seq_len)
        self.attention = nn.Parameter(torch.ones([int(backbone_cfg.num_mixers)+1, 1]))
        self.ImageCls = ImageCls(classification_cfg)

    def forward(self, inputs):
        roll_step = [1, 1, 2, 2, 3, 3]
        in_features = self.patch_embedding(inputs)
        stages = in_features.unsqueeze(1)
        for i, mixer in enumerate(self.mixers.mixers):
            roll_features = torch.roll(in_features, (roll_step[i], roll_step[i]), (1, 2))
            in_features = mixer(in_features+roll_features)
            stages = torch.cat((stages, in_features.unsqueeze(1)), dim=1)
        in_features = torch.sum(torch.matmul(stages.transpose(1, -1), self.attention), dim=-1).transpose(1,-1)
        outs = self.ImageCls(in_features)
        return outs


class TCAMixerNLPCls(nn.Module):
    def __init__(self, mode, model_cfg: DictConfig, dataset_cfg: DictConfig, **kwargs):
        super(TCAMixerNLPCls, self).__init__(**kwargs)
        self.pipeline = nn.Sequential(
            nn.Linear((2 * model_cfg.bottleneck.window_size + 1) * model_cfg.bottleneck.feature_size,
                      model_cfg.bottleneck.hidden_dim),
            TCAMixer(mode, backbone_cfg=model_cfg.backbone, max_seq_len=dataset_cfg.dataset_type.max_seq_len))
        self.ImageCls = ImageCls(dataset_cfg.classification)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.size(1) == 2:
            sentence_first, sentence_second = torch.chunk(inputs, 2, dim=1)
            # print(sentence_second.shape, sentence_first.shape)
            sentence_first = sentence_first.squeeze(1)
            sentence_second = sentence_second.squeeze(1)
            u = self.pipeline(sentence_first)
            v = self.pipeline(sentence_second)
            return self.ImageCls(torch.cat((u, v, torch.abs(u - v)), dim=1))
        else:
            reprs = self.pipeline(inputs)
        return self.ImageCls(reprs)

