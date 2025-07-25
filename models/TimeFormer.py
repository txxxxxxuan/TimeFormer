import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import MaskAttention, AttentionLayer
from layers.Embed import PatchDidEmbedding
import numpy as np


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):

        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


def Hawkes_matrix(T, alpha=.9):
    matrix = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1):
            matrix[i, j] = alpha ** (i - j)
    return torch.from_numpy(matrix)


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=10, stride=10):

        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.down_sampling_layers = configs.down_sampling_layers

        self.sqrt_values = []
        self.head_nf = []
        for i in range(configs.down_sampling_layers + 1):
            sqrt_value = int(torch.ceil(torch.sqrt(torch.tensor(self.seq_len // (i + 1)))).item())
            self.head_nf.append(configs.d_model * sqrt_value)
            self.sqrt_values.append(sqrt_value)

        self.patch_embedding = nn.ModuleList([PatchDidEmbedding(configs,
                                                                configs.d_model, self.sqrt_values[i],
                                                                self.sqrt_values[i],
                                                                self.sqrt_values[i],
                                                                configs.dropout) for i in
                                              range(configs.down_sampling_layers + 1)])

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        self.intraencoder = nn.ModuleList([Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        MaskAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False, beta=configs.beta), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
        ) for i in range(configs.down_sampling_layers + 1)])


        self.interencoder = nn.ModuleList([Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        MaskAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False, beta=configs.beta), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
        ) for i in range(configs.down_sampling_layers + 1)])

        self.projection = nn.ModuleList(
            [nn.Linear(self.head_nf[i], configs.d_model) for i in range(configs.down_sampling_layers + 1)])

        self.projection_layer = nn.Linear(
            configs.pred_len, configs.pred_len, bias=True)

        self.head = nn.ModuleList(
            [FlattenHead(self.head_nf[i], configs.pred_len,
                         head_dropout=configs.dropout) for i in range(configs.down_sampling_layers + 1)])

    def __multi_scale_process_inputs(self, x_enc):
        down_sampling_window = []
        if self.down_sampling_layers == 0:
            return [x_enc]
        if self.down_sampling_layers == 1:
            down_sampling_window = [2]
        if self.down_sampling_layers == 2:
            down_sampling_window = [2, 3]
        if self.down_sampling_layers == 3:
            down_sampling_window = [2, 3, 4]

        down_pool = nn.ModuleList()
        for x in down_sampling_window:
            down_pool.append(torch.nn.AvgPool1d(x))

        x_enc = x_enc.permute(0, 2, 1)

        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool[i](x_enc)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))

        return x_enc_sampling_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, T, N = x_enc.shape

        x_enc_sampling_list = self.__multi_scale_process_inputs(x_enc)
        dec_out_list = []
        for i in range(self.configs.down_sampling_layers + 1):
            x_enc = self.normalize_layers[i](x_enc_sampling_list[i], 'norm')

            x_enc = x_enc.permute(0, 2, 1)

            enc_out, B, n_vars, sqrt_value = self.patch_embedding[i](x_enc)
            enc_out, m1 = self.intraencoder[i](enc_out)

            enc_out = self.projection[i](enc_out.reshape(-1, enc_out.shape[-1] * enc_out.shape[-2])).reshape(B * n_vars,
                                                                                                             sqrt_value,
                                                                                                             -1)
            enc_out, m2 = self.interencoder[i](enc_out)
            enc_out = torch.reshape(enc_out, (-1, N, enc_out.shape[-2], enc_out.shape[-1]))
            enc_out = self.head[i](enc_out).transpose(-1, -2)
            dec_out_list.append(enc_out)

        dec_out = torch.stack(dec_out_list, dim=-1).mean(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')

        return dec_out
