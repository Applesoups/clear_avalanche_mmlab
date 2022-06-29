import numpy as np

import torch
import torch.nn as nn
from mmcv.runner.base_module import ModuleList
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.cnn.bricks.transformer import BaseModule
from mmcls.models.builder import BACKBONES
from mmcls.models import VisionTransformer
from mmcls.models.backbones.vision_transformer import TransformerEncoderLayer


@BACKBONES.register_module()
class AdaptFormer(VisionTransformer):
    def __init__(self,
                 arch='base',
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 use_adapt=True,
                 adapt_bottleneck_dim=64,
                 drop_rate=0.,
                 adapt_bottleneck_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 with_cls_token=True,
                 output_cls_token=True,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None):
        super(AdaptFormer, self).__init__(
            arch,
            img_size,
            patch_size,
            in_channels,
            out_indices,
            drop_rate,
            drop_path_rate,
            qkv_bias,
            norm_cfg,
            final_norm,
            with_cls_token,
            output_cls_token,
            interpolate_mode,
            patch_cfg,
            layer_cfgs,
            init_cfg)

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.arch_settings['feedforward_channels'],
                use_adapt=use_adapt,
                bottleneck_dim=adapt_bottleneck_dim,
                drop_rate=drop_rate,
                bottleneck_drop_rate=adapt_bottleneck_drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(Transformer(**_layer_cfg))


class Transformer(TransformerEncoderLayer):
    def __init__(self,
                 embed_dims,
                 bottleneck_dim,
                 bottleneck_drop_rate=0.,
                 use_adapt=True,
                 **kwargs):
        super(Transformer, self).__init__(embed_dims=embed_dims, **kwargs)

        # add AdaptMLP
        self.use_adapt = use_adapt
        if use_adapt:
            self.adapter = AdaptMLP(
                embed_dims=embed_dims,
                bottleneck_dim=bottleneck_dim,
                drop_rate=bottleneck_drop_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        if self.use_adapt:
            adapt_x = self.adapter(x, identity=0)
        else:
            adapt_x = 0
        res = x
        x = self.ffn(self.norm2(x), identity=adapt_x)
        return x + res


class AdaptMLP(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 bottleneck_dim=64,
                 drop_rate=0.,
                 adapt_scalar=1.,
                 learnable_scale=True,
                 add_identity=True,
                 init_cfg=None):
        super(AdaptMLP, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.bottleneck_dim = bottleneck_dim
        self.add_identity = add_identity

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(adapt_scalar))
        else:
            self.scale = adapt_scalar

        self.down_proj = nn.Linear(embed_dims, bottleneck_dim)
        self.relu = build_activation_layer(dict(type='ReLU'))
        self.dropout = nn.Dropout(p=drop_rate)
        self.up_proj = nn.Linear(bottleneck_dim, embed_dims)
        self.ln = nn.LayerNorm(embed_dims)

    def forward(self, x, identity=None):
        down = self.down_proj(x)
        down = self.relu(down)
        down = self.dropout(down)

        up = self.up_proj(down)
        up = up * self.scale
        up = self.ln(up)

        if not self.add_identity:
            return up
        if identity is None:
            identity = x
        return up + identity
