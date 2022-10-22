from omegaconf import DictConfig
from math import sqrt
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange


class Bernolli_sampling_nlp:
    def __init__(self, max_len=100, prob=1):
        self.max_len = max_len
        self.prob = prob

    def __call__(self, inputs):
        n_samples, max_seq_len, embedding_dim = inputs.size(0), inputs.size(1), inputs.size(-1)
        Benoulli = torch.distributions.bernoulli.Bernoulli(self.prob)
        masks = Benoulli.sample((n_samples, max_seq_len)).unsqueeze(-1).repeat(1, 1, embedding_dim).bool().cuda()
        inputs = F.softmax(inputs.masked_fill(~masks, -np.inf), dim=-2)
        return inputs


class Bernolli_sampling_cv:
    def __init__(self, max_len=100, prob=1):
        self.max_len = max_len
        self.prob = prob

    def __call__(self, inputs):
        n_samples, max_seq_len, embedding_dim = inputs.size(0), inputs.size(1), inputs.size(-1)
        Benoulli = torch.distributions.bernoulli.Bernoulli(self.prob)
        masks = Benoulli.sample((n_samples, max_seq_len, embedding_dim)).cuda().bool()
        inputs = F.softmax(inputs.masked_fill(~masks, -np.inf), dim=-2)
        return inputs


class TCA(nn.Module):
    def __init__(self, mode, max_seq_len, embedding_dim, prob, kernel_size, dilation, padding):
        super(TCA, self).__init__()
        self.embedding_dim = embedding_dim
        self.local_information = nn.Conv1d(embedding_dim, embedding_dim,
                                                          kernel_size, 1, padding, dilation, groups=embedding_dim)
        self.activate = nn.GELU()
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.global_information = nn.Linear(max_seq_len, max_seq_len)
        self.bernolli_sampling = self.Choice_Bernolli(mode)(max_len=max_seq_len, prob=prob)
        self.softmax = nn.Softmax(-1)

    def Choice_Bernolli(self, mode: str):
        if mode == "cv":
            return Bernolli_sampling_cv
        else:
            return Bernolli_sampling_nlp

    def forward(self, x):
        x = x.transpose(1, 2)
        # [N, embedding_dim 4, max_seq_len 384]
        q = self.bn(self.activate(self.local_information(x)+x))
        k = self.activate(self.bernolli_sampling(x))
        v = self.activate(self.global_information(x))
        attention = self.softmax(torch.bmm(q, k.transpose(1, 2))/sqrt(self.embedding_dim))
        output = torch.bmm(attention, v)
        return output.transpose(1, 2), attention


class MHTCA(nn.Module):
    def __init__(self, n_head, mode, max_seq_len, embedding_dim, prob, kernel_size, dilation, padding):
        super(MHTCA, self).__init__()
        assert max_seq_len % n_head == 0, 'max_seq_len must be divisible by the n_head.'
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        self.input_dim = int(max_seq_len // n_head)
        # print(self.embedding_dim, self.n_head, self.max_seq_len, self.input_dim)
        self.local_information = nn.Conv1d(embedding_dim, embedding_dim,
                                                          kernel_size, 1, padding, dilation, self.input_dim)
        self.activate = nn.GELU()
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.global_information = nn.Linear(self.input_dim, self.input_dim)
        self.mode = mode
        if mode == "cv":
            self.bernolli_sampling = Bernolli_sampling_cv(prob=prob)
        else:
            self.bernolli_sampling = Bernolli_sampling_nlp(prob=prob)
        self.softmax = nn.Softmax(-1)
        self.trans = Rearrange("b (m h) d -> (b h) d m ", h=n_head)
        self.trans2 = Rearrange("(b h) d m  -> b (m h) d ", h=n_head)

    def forward(self, inputs):
        #  b (chw) (p1 p2)
        # print(inputs.shape)
        if self.mode == "cv":
            q = self.trans(inputs)
            k = self.trans(inputs)
            v = self.trans(inputs)
        else:
            q = inputs.view(-1, self.embedding_dim, self.input_dim)
            k = inputs.view(-1, self.embedding_dim, self.input_dim)
            v = inputs.view(-1, self.embedding_dim, self.input_dim)
        q = self.bn(self.activate(self.local_information(q)+q))
        k = self.activate(self.bernolli_sampling(k))
        v = self.activate(self.global_information(v))
        # print(q.shape, k.shape, v.shape)
        attention = self.softmax(torch.bmm(q, k.transpose(1, 2)) / sqrt(self.embedding_dim))
        output = torch.bmm(attention, v)
        if self.mode == "cv":
            return self.trans2(output), attention
        else:
            return output.reshape(-1, self.max_seq_len, self.embedding_dim), attention


class TCAMixer(nn.Module):

    def __init__(self, mode, backbone_cfg: DictConfig, max_seq_len: int, **kwargs):
        hidden_dim = backbone_cfg["hidden_dim"]
        index = backbone_cfg["index"]
        kernel_size = backbone_cfg["kernel_size"]
        dilation = backbone_cfg["dilation"]
        padding = backbone_cfg["padding"]
        n_heads = backbone_cfg["num_heads"]
        num_mixers = backbone_cfg["num_mixers"]
        super(TCAMixer, self).__init__(**kwargs)
        self.mixers = nn.Sequential(
            *[MixerLayer(n_heads, mode, max_seq_len, hidden_dim, index, kernel_size[i], dilation[i], padding[i], **kwargs)
                                  for i in range(num_mixers)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mixers(inputs)


class MixerLayer(nn.Module):

    def __init__(self, n_heads, mode, max_seq_len: int, hidden_dim: int, index: int, kernel_size: int,
                 dilation: int, padding: int, **kwargs):
        super(MixerLayer, self).__init__(**kwargs)
        self.kernel_size, self.dilation, self.padding = kernel_size, dilation, padding
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        # attention = attention_choice(index)
        self.sa = MHTCA(n_heads, 'nlp', max_seq_len, hidden_dim, 0.8, kernel_size, dilation, padding)
        # self.sa = TCA("nlp", max_seq_len, hidden_dim, 0.8, kernel_size, dilation, padding)
        self.activate = nn.GELU()
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.mlp_2 = MlpLayer(hidden_dim, hidden_dim)

    def forward(self, inputs) -> torch.Tensor:
        # print(self.kernel_size, self.dilation, self.padding)
        residual = inputs
        outputs = self.layer_norm_1(inputs)
        outputs, attention = self.sa(outputs)
        # print(outputs[0])
        outputs = self.activate(outputs + residual)
        residual = outputs
        outputs = self.layer_norm_2(outputs)
        outputs = self.activate(self.mlp_2(self.dropout(outputs)) + residual)
        return outputs


class MlpLayer(nn.Module):

    def __init__(self, hidden_dim: int, intermediate_dim: int, **kwargs):
        super(MlpLayer, self).__init__(**kwargs)
        self.layers = nn.Sequential(*[
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim)
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


if __name__ == "__main__":
    mha = MHTCA(7, "cv", 588, 256, 0.6, 3, 1, 1).cuda()
    a = torch.randn([5, 588, 256]).cuda()
    b = mha(a)
    print(b)
