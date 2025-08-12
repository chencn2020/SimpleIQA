"""TOP-IQ metric, proposed by

TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment.
Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, Weisi Lin.
Transactions on Image Processing, 2024.

Paper link: https://arxiv.org/abs/2308.03060

"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import timm

import copy
from .topiq_swin import create_swin

import warnings

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation='gelu',
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, src):
        src2 = self.norm1(src)
        q = k = src2
        src2, self.attn_map = self.self_attn(q, k, value=src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation='gelu',
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, tgt, memory):
        memory = self.norm2(memory)
        tgt2 = self.norm1(tgt)
        tgt2, self.attn_map = self.multihead_attn(query=tgt2, key=memory, value=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src):
        output = src

        for layer in self.layers:
            output = layer(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory)

        return output


class GatedConv(nn.Module):
    def __init__(self, weightdim, ksz=3):
        super().__init__()

        self.splitconv = nn.Conv2d(weightdim, weightdim * 2, 1, 1, 0)
        self.act = nn.GELU()

        self.weight_blk = nn.Sequential(
            nn.Conv2d(weightdim, 64, 1, stride=1),
            nn.GELU(),
            nn.Conv2d(64, 64, ksz, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, ksz, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1, x2 = self.splitconv(x).chunk(2, dim=1)
        weight = self.weight_blk(x2)
        x1 = self.act(x1)
        return x1 * weight


class CFANet(nn.Module):
    def __init__(
        self,
        semantic_model_name='resnet50',
        model_name='cfanet_nr_koniq_res50',
        backbone_pretrain=True,
        in_size=None,
        use_ref=False,
        num_class=1,
        num_crop=1,
        crop_size=256,
        inter_dim=256,
        num_heads=4,
        num_attn_layers=1,
        dprate=0.1,
        activation='gelu',
        pretrained=True,
        out_act=False,
        block_pool='weighted_avg',
        test_img_size=None,
        align_crop_face=True,
    ):
        super().__init__()

        self.in_size = in_size

        self.model_name = model_name
        self.semantic_model_name = semantic_model_name
        self.semantic_level = -1
        self.crop_size = crop_size
        self.use_ref = use_ref

        self.num_class = num_class
        self.block_pool = block_pool
        self.test_img_size = test_img_size

        self.align_crop_face = align_crop_face

        # =============================================================
        # define semantic backbone network
        # =============================================================

        if 'swin' in semantic_model_name:
            self.semantic_model = create_swin(
                semantic_model_name, pretrained=True, drop_path_rate=0.0
            )
            feature_dim = self.semantic_model.num_features
            feature_dim_list = [
                int(self.semantic_model.embed_dim * 2**i)
                for i in range(self.semantic_model.num_layers)
            ]
            feature_dim_list = feature_dim_list[1:] + [feature_dim]
            all_feature_dim = sum(feature_dim_list)
        else:
            self.semantic_model = timm.create_model(
                semantic_model_name, pretrained=backbone_pretrain, features_only=True
            )
            feature_dim_list = self.semantic_model.feature_info.channels()
            feature_dim = feature_dim_list[self.semantic_level]
            all_feature_dim = sum(feature_dim_list)
            self.fix_bn(self.semantic_model)


        # =============================================================
        # define self-attention and cross scale attention blocks
        # =============================================================

        self.fusion_mul = 3 if use_ref else 1
        ca_layers = sa_layers = num_attn_layers

        self.act_layer = nn.GELU() if activation == 'gelu' else nn.ReLU()
        dim_feedforward = min(4 * inter_dim, 2048)

        # gated local pooling and self-attention
        tmp_layer = TransformerEncoderLayer(
            inter_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=dprate,
            activation=activation,
        )
        self.sa_attn_blks = nn.ModuleList()
        self.dim_reduce = nn.ModuleList()
        self.weight_pool = nn.ModuleList()
        for idx, dim in enumerate(feature_dim_list):
            dim = dim * 3 if use_ref else dim
            if use_ref:
                self.weight_pool.append(
                    nn.Sequential(
                        nn.Conv2d(dim // 3, 64, 1, stride=1),
                        self.act_layer,
                        nn.Conv2d(64, 64, 3, stride=1, padding=1),
                        self.act_layer,
                        nn.Conv2d(64, 1, 3, stride=1, padding=1),
                        nn.Sigmoid(),
                    )
                )
            else:
                self.weight_pool.append(GatedConv(dim))

            self.dim_reduce.append(
                nn.Sequential(
                    nn.Conv2d(dim, inter_dim, 1, 1),
                    self.act_layer,
                )
            )

            self.sa_attn_blks.append(TransformerEncoder(tmp_layer, sa_layers))

        # cross scale attention
        self.attn_blks = nn.ModuleList()
        tmp_layer = TransformerDecoderLayer(
            inter_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=dprate,
            activation=activation,
        )
        for i in range(len(feature_dim_list) - 1):
            self.attn_blks.append(TransformerDecoder(tmp_layer, ca_layers))

        # attention pooling and MLP layers
        self.attn_pool = TransformerEncoderLayer(
            inter_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=dprate,
            activation=activation,
        )

        linear_dim = inter_dim
        self.score_linear = [
            nn.LayerNorm(linear_dim),
            nn.Linear(linear_dim, linear_dim),
            self.act_layer,
            nn.LayerNorm(linear_dim),
            nn.Linear(linear_dim, linear_dim),
            self.act_layer,
            nn.Linear(linear_dim, self.num_class),
        ]

        # make sure output is positive, useful for 2AFC datasets with probability labels
        if out_act and self.num_class == 1:
            self.score_linear.append(nn.Softplus())

        if self.num_class > 1:
            self.score_linear.append(nn.Softmax(dim=-1))

        self.score_linear = nn.Sequential(*self.score_linear)

        self.h_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 32, 1))
        self.w_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 1, 32))

        nn.init.trunc_normal_(self.h_emb.data, std=0.02)
        nn.init.trunc_normal_(self.w_emb.data, std=0.02)
        self._init_linear(self.dim_reduce)
        self._init_linear(self.sa_attn_blks)
        self._init_linear(self.attn_blks)
        self._init_linear(self.attn_pool)

        self.eps = 1e-8
        self.crops = num_crop

    def _init_linear(self, m):
        for module in m.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                nn.init.constant_(module.bias.data, 0)

    def fix_bn(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False
                m.eval()

    def get_swin_feature(self, model, x):
        b, c, h, w = x.shape
        x = model.patch_embed(x)
        if model.absolute_pos_embed is not None:
            x = x + model.absolute_pos_embed
        x = model.pos_drop(x)
        feat_list = []
        for ly in model.layers:
            x = ly(x)
            feat_list.append(x)

        h, w = h // 8, w // 8
        for idx, f in enumerate(feat_list):
            feat_list[idx] = f.transpose(1, 2).reshape(b, f.shape[-1], h, w)
            if idx < len(feat_list) - 2:
                h, w = h // 2, w // 2

        return feat_list

    def dist_func(self, x, y, eps=1e-12):
        return torch.sqrt((x - y) ** 2 + eps)

    def forward_cross_attention(self, x, y=None):
        if 'swin' in self.semantic_model_name:
            dist_feat_list = self.get_swin_feature(self.semantic_model, x)
            if self.use_ref:
                ref_feat_list = self.get_swin_feature(self.semantic_model, y)
            self.semantic_model.eval()
        elif 'clip' in self.semantic_model_name:
            visual_model = self.semantic_model[0].visual.to(x.device)
            dist_feat_list = visual_model.forward_features(x)
            if self.use_ref:
                ref_feat_list = visual_model.forward_features(y)
        else:
            dist_feat_list = self.semantic_model(x)
            if self.use_ref:
                ref_feat_list = self.semantic_model(y)
            self.fix_bn(self.semantic_model)
            self.semantic_model.eval()

        start_level = 0
        end_level = len(dist_feat_list)

        b, c, th, tw = dist_feat_list[end_level - 1].shape
        pos_emb = torch.cat(
            (
                self.h_emb.repeat(1, 1, 1, self.w_emb.shape[3]),
                self.w_emb.repeat(1, 1, self.h_emb.shape[2], 1),
            ),
            dim=1,
        )

        token_feat_list = []
        for i in reversed(range(start_level, end_level)):
            tmp_dist_feat = dist_feat_list[i]

            # gated local pooling
            if self.use_ref:
                tmp_ref_feat = ref_feat_list[i]
                diff = self.dist_func(tmp_dist_feat, tmp_ref_feat)

                tmp_feat = torch.cat([tmp_dist_feat, tmp_ref_feat, diff], dim=1)
                weight = self.weight_pool[i](diff)
                tmp_feat = tmp_feat * weight
            else:
                tmp_feat = self.weight_pool[i](tmp_dist_feat)

            if tmp_feat.shape[2] > th and tmp_feat.shape[3] > tw:
                tmp_feat = F.adaptive_avg_pool2d(tmp_feat, (th, tw))

            # self attention
            tmp_pos_emb = F.interpolate(
                pos_emb, size=tmp_feat.shape[2:], mode='bicubic', align_corners=False
            )
            tmp_pos_emb = tmp_pos_emb.flatten(2).permute(2, 0, 1)

            tmp_feat = self.dim_reduce[i](tmp_feat)
            tmp_feat = tmp_feat.flatten(2).permute(2, 0, 1)
            tmp_feat = tmp_feat + tmp_pos_emb

            tmp_feat = self.sa_attn_blks[i](tmp_feat)
            token_feat_list.append(tmp_feat)

        # high level -> low level: coarse to fine
        query = token_feat_list[0]
        query_list = [query]
        for i in range(len(token_feat_list) - 1):
            key_value = token_feat_list[i + 1]
            query = self.attn_blks[i](query, key_value)
            query_list.append(query)

        final_feat = self.attn_pool(query)
        out_score = self.score_linear(final_feat.mean(dim=0))

        return out_score

    def preprocess_face(self, x):
        warnings.warn(
            f'The faces will be aligned, cropped and resized to 512x512 with facexlib. Currently, this metric does not support batch size > 1 and gradient backpropagation.',
            UserWarning,
        )
        # warning message
        device = x.device
        assert x.shape[0] == 1, f'Only support batch size 1, but got {x.shape[0]}'
        self.face_helper.clean_all()
        self.face_helper.input_img = x[0].permute(1, 2, 0).cpu().numpy() * 255
        self.face_helper.input_img = self.face_helper.input_img[..., ::-1]
        if (
            self.face_helper.get_face_landmarks_5(
                only_center_face=True, eye_dist_threshold=5
            )
            > 0
        ):
            self.face_helper.align_warp_face()
            x = self.face_helper.cropped_faces[0]
            x = (
                torch.from_numpy(x[..., ::-1].copy())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                / 255.0
            )
            return x.to(device)
        else:
            assert False, f'No face detected in the input image.'

    def forward(self, x, y=None, return_mos=True, return_dist=False):
        if self.use_ref:
            assert y is not None, f'Please input y when use reference is True.'
        else:
            y = None

        score = self.forward_cross_attention(x, y)

        mos = dist_to_mos(score)

        return mos
    
def dist_to_mos(dist_score: torch.Tensor) -> torch.Tensor:
    """
    Convert distribution prediction to MOS score.
    For datasets with detailed score labels, such as AVA.

    Args:
        dist_score (torch.Tensor): (*, C), C is the class number.

    Returns:
        torch.Tensor: (*, 1) MOS score.
    """
    num_classes = dist_score.shape[-1]
    mos_score = dist_score * torch.arange(1, num_classes + 1).to(dist_score)
    mos_score = mos_score.sum(dim=-1, keepdim=True)
    return mos_score