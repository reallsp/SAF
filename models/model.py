import torch.nn as nn
from .resnet import resnet50
from .bert import Bert
import torchvision.models as models
import torch
import os
import cv2
import math
from .bi_lstm import BiLSTM
import torch.nn.functional as F
import pickle
import numpy as np
from .modeling import VisionTransformer, CONFIGS


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * self.num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, input_masks=None):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if input_masks is not None:
            extended_attention_mask = input_masks.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(
                dtype=torch.float32
            )  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attn = attn + extended_attention_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C, -1).permute(3, 0, 1, 2)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=gelu,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, input_masks=None):
        x, attn = self.attn(self.norm1(x), input_masks)
        ans = ()
        for i in range(x.shape[0]):
            ans = ans + (x[i],)
        return ans, attn


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
                if args.img_model == "vit":
            config = CONFIGS[args.model_type]
            self.image_model = VisionTransformer(config)
            self.image_model.load_from(np.load(args.pretrain_dir))
            self.language_model = Bert()
            inp_size = 768
        elif args.img_model == "resnet50":
            self.image_model = resnet50(pretrained=True)
            inp_size = 2048
        if args.text_model == "bert":
            self.language_model = Bert()
            text_size = 768
        elif args.text_model == "bilstm":
            self.language_model = BiLSTM(args)
            self.language_model.apply(self.language_model.weight_init)
            text_size = 1024

        self.block = Block(
            dim=768,
            num_heads=args.num_heads,
            mlp_ratio=0.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
        )

        self.bottleneck_image = nn.BatchNorm1d(args.feature_size)
        self.bottleneck_image.bias.requires_grad_(False)
        self.bottleneck_image.apply(weights_init_kaiming)
        self.bottleneck_text = nn.BatchNorm1d(args.feature_size)
        self.bottleneck_text.bias.requires_grad_(False)
        self.bottleneck_text.apply(weights_init_kaiming)
        self.fc = nn.Linear(args.num_heads * args.feature_size, args.feature_size)

    def forward(self, images, tokens, segments, input_masks):
        image_feats = self.image_model(images)[0]
        text_feats = self.language_model(tokens, segments, input_masks)[0]
        img_global = self.bottleneck_image(image_feats[:, 0, :])
        text_global = self.bottleneck_image(text_feats[:, 0, :])
        img_output = (img_global,)
        text_output = (text_global,)
        image_parts, image_attn = self.block(image_feats)
        text_parts, text_attn = self.block(text_feats, input_masks)
        for j in range(len(image_parts)):
            img_output = img_output + (self.bottleneck_image(image_parts[j][:, 0, :]),)
            text_output = text_output + (self.bottleneck_image(text_parts[j][:, 0, :]),)
        img_f = torch.stack(img_output, dim=1)
        text_f = torch.stack(text_output, dim=1)
        return img_output, text_output, image_attn, text_attn, img_f, text_f
