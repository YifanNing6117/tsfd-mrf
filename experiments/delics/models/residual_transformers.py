# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from scipy import ndimage
from . import transformer_configs as configs
from .residual_swin_transformers import SwinIR
from .residual_swin_transformers_multi_modal import Swin_multi_modal_block

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size##paraphrase

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3,input_dim=3,old = 1):
        super(Embeddings, self).__init__()
        self.config = config
        img_size = _pair(img_size)
        grid_size = config.patches["grid"]
        patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
        in_channels = 1024
        #Learnable patch embeddings
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        #learnable positional encodings
        self.positional_encoding = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.positional_encoding
        embeddings = self.dropout(embeddings)
        return embeddings

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self,config, img_size, vis,in_channels=3,old = 1):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config,img_size=img_size,input_dim=in_channels,old = old)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,dim2=None):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        #use_dropout= use_dropo
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ART_block(nn.Module):
    def __init__(self,config, input_dim, img_size=224,transformer = None):
        super(ART_block, self).__init__()
        self.transformer = transformer
        self.config = config
        ngf = 72*3
        mult = 4
        use_bias = False
        norm_layer = nn.BatchNorm2d
        padding_type = 'reflect'
        if self.transformer:
            # Downsample
            model = [nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3,
                               stride=2, padding=1, bias=use_bias),
                     norm_layer(ngf * 8),
                     nn.ReLU(True)]
            model += [nn.Conv2d(ngf * 8, 1024, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(1024),
                      nn.ReLU(True)]
            setattr(self, 'downsample', nn.Sequential(*model))
            #Patch embedings
            self.embeddings = Embeddings(config, img_size=img_size, input_dim=input_dim)
            # Upsampling block
            model = [nn.ConvTranspose2d(self.config.hidden_size, ngf * 8,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                     norm_layer(ngf * 8),
                     nn.ReLU(True)]
            model += [nn.ConvTranspose2d(ngf * 8, ngf * 4,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(ngf * 4),
                      nn.ReLU(True)]
            setattr(self, 'upsample', nn.Sequential(*model))
            #Channel compression
            self.cc = channel_compression(ngf * 8, ngf * 4)
        # Residual CNN
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                             use_bias=use_bias)]
        setattr(self, 'residual_cnn', nn.Sequential(*model))

    def forward(self, x):
        if self.transformer:
            # downsample
            down_sampled = self.downsample(x)
            # embed
            embedding_output = self.embeddings(down_sampled)
            # feed to transformer
            transformer_out, attn_weights = self.transformer(embedding_output)
            B, n_patch, hidden = transformer_out.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
            h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
            transformer_out = transformer_out.permute(0, 2, 1)
            transformer_out = transformer_out.contiguous().view(B, hidden, h, w)
            # upsample transformer output
            transformer_out = self.upsample(transformer_out)
            # concat transformer output and resnet output
            x = torch.cat([transformer_out, x], dim=1)
            # channel compression
            x = self.cc(x)
        # residual CNN
        x = self.residual_cnn(x)
        return x

########################################################################################################
########################################################################################################

from .residual_swin_transformers import Swin_block
# from .residual_swin_transformers import SwinIR

########################################################################################################
########################################################################################################

class CNN_block(nn.Module):
    def __init__(self, input_dim=256, img_size=256):
        super(CNN_block, self).__init__()

        use_bias = False
        norm_layer = nn.BatchNorm2d #nn.InstanceNorm2d 
        padding_type = 'reflect'

        # Residual CNN
        model = [ResnetBlock(dim=input_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)]
        setattr(self, 'cnn', nn.Sequential(*model))

    def forward(self, x):  # torch.Size([batch_size, 256, 64, 64])
        x = self.cnn(x)  # torch.Size([batch_size, 256, 64, 64])
        return x

########################################################################################################
########################################################################################################



########Generator############
class ResViT_new(nn.Module):
# class ResViT(nn.Module):
    def __init__(self,config, input_dim, img_size=224, output_dim=3, vis=False):
#        super(ResViT, self).__init__()
        super(ResViT_new, self).__init__()
        self.transformer_encoder = Encoder(config, vis)
        self.config = config
        output_nc = output_dim
        ngf = 64
        use_bias = False
        norm_layer = nn.BatchNorm2d
        padding_type = 'reflect'
        mult = 4

        ############################################################################################
        # Layer1-Encoder1
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_dim, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        setattr(self, 'encoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer2-Encoder2
        n_downsampling = 2
        model = []
        i = 0
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer3-Encoder3
        model = []
        i = 1
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_3', nn.Sequential(*model))
        ####################################ART Blocks##############################################
        mult = 4
        self.art_1 = ART_block(self.config, input_dim, img_size,transformer = self.transformer_encoder)
        self.art_2 = ART_block(self.config, input_dim, img_size,transformer = None)
        self.art_3 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_4 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_5 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_6 = ART_block(self.config, input_dim, img_size,transformer = self.transformer_encoder)
        self.art_7 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_8 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_9 = ART_block(self.config, input_dim, img_size, transformer=None)
        ############################################################################################
        # Layer13-Decoder1
        n_downsampling = 2
        i = 0
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer14-Decoder2
        i = 1
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer15-Decoder3
        model = []
        model = [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_dim, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        setattr(self, 'decoder_3', nn.Sequential(*model))

    ############################################################################################
    ##################### Added by sxb #########################################################

        self.swin_block_1 = Swin_block(in_chans=256, embed_dim = 256, patch_size=1, img_size=64, transformer=True)  #Swin_block(img_size=64, patch_size=16, in_chans=256, embed_dim=512)
        self.cnn_block_11 = CNN_block(input_dim=256, img_size=64)
        self.cnn_block_12 = CNN_block(input_dim=256, img_size=64)


        self.swin_block_2 = Swin_block(in_chans=256, embed_dim = 256, patch_size=1, img_size=64, transformer=True) #Swin_block(img_size=64, patch_size=16, in_chans=256, embed_dim=512)
        self.cnn_block_21 = CNN_block(input_dim=256, img_size=64)
        self.cnn_block_22 = CNN_block(input_dim=256, img_size=64)


        self.swin_block_3 = Swin_block(in_chans=256, embed_dim = 256, patch_size=1, img_size=64, transformer=True)
        self.cnn_block_31 = CNN_block(input_dim=256, img_size=64)
        self.cnn_block_32 = CNN_block(input_dim=256, img_size=64)

        self.swin_block_4 = Swin_block(in_chans=256, embed_dim = 256, patch_size=1, img_size=64, transformer=True)
        self.cnn_block_41 = CNN_block(input_dim=256, img_size=64)
        self.cnn_block_42 = CNN_block(input_dim=256, img_size=64)

        self.swin_block_5 = Swin_block(in_chans=256, embed_dim = 256, patch_size=1, img_size=64, transformer=True)
        self.cnn_block_51 = CNN_block(input_dim=256, img_size=64)
        self.cnn_block_52 = CNN_block(input_dim=256, img_size=64)

    #############################################################################################
    #############################################################################################

    def forward(self, x):

        #print('\033[1;31m---residual_transformers.py---\033[0m')
        #print("x.shape = ", x.shape)
        # x.shape =  torch.Size([1, 1, 256, 256])

        # Pass input through cnn encoder of ResViT
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.encoder_3(x)

        # print('\033[1;31m---residual_transformers.py---\033[0m')
        # print("x.shape = ", x.shape)
        # x.shape =  torch.Size([1, 256, 64, 64])

    #############################################################################################
    #############################################################################################

        x = self.cnn_block_11(x)
        x = self.cnn_block_12(x)
        x = self.swin_block_1(x)
        # print('\033[1;31m---residual_transformers.py---\033[0m')
        # print("x_test.shape = ", x_test.shape)
        # x_test.shape =  torch.Size([1, 256, 64, 64])

        x = self.cnn_block_21(x)
        x = self.cnn_block_22(x)
        x = self.swin_block_2(x)

        x = self.cnn_block_31(x)
        x = self.cnn_block_32(x)
        x = self.swin_block_3(x)

        x = self.cnn_block_41(x)
        x = self.cnn_block_42(x)
        x = self.swin_block_4(x)

        x = self.cnn_block_51(x)
        x = self.cnn_block_52(x)
        x = self.swin_block_5(x)

    #############################################################################################
    #############################################################################################


        #Information Bottleneck
#        x = self.art_1(x)
#        x = self.art_2(x)
#        x = self.art_3(x)
#        x = self.art_4(x)
#        x = self.art_5(x)
#        x = self.art_6(x)
#        x = self.art_7(x)
#        x = self.art_8(x)
#        x = self.art_9(x)

        # print('\033[1;31m---residual_transformers.py---\033[0m')
        # print("x.shape = ", x.shape)
        # x.shape =  torch.Size([1, 256, 64, 64])

        # decoder
        x = self.decoder_1(x)
        # print('\033[1;31m---residual_transformers.py---\033[0m')
        # print("x.shape = ", x.shape)
        x = self.decoder_2(x)
        # print('\033[1;31m---residual_transformers.py---\033[0m')
        # print("x2.shape = ", x.shape)
        x = self.decoder_3(x)
        # print('\033[1;31m---residual_transformers.py---\033[0m')
        # print("x3.shape = ", x.shape)


        return x


    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            if self.config.name == 'b16':
                self.art_1.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
                self.art_1.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

                self.art_6.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
                self.art_6.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer_encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer_encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.art_1.embeddings.positional_encoding
            if posemb.size() == posemb_new.size():
                self.art_1.embeddings.positional_encoding.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.art_1.embeddings.positional_encoding1.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.art_1.embeddings.positional_encoding.copy_(np2th(posemb))

            #############
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.art_6.embeddings.positional_encoding
            if posemb.size() == posemb_new.size():
                self.art_6.embeddings.positional_encoding.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.art_6.embeddings.positional_encoding.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.art_6.embeddings.positional_encoding.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer_encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)


# class Res_CNN(nn.Module):
class ResViT(nn.Module):
    def __init__(self, config, input_dim, img_size=224, output_dim=3, vis=False):
#        super(Res_CNN, self).__init__()
        super(ResViT, self).__init__()
        self.config = config
        output_nc = output_dim
        ngf = 72*3
        use_bias = False
        norm_layer = nn.BatchNorm2d
        padding_type = 'reflect'
        mult = 4
        # print(input_dim)
        # print(output_dim)
        ############################################################################################
        # Layer1-Encoder1
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_dim, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        setattr(self, 'encoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer2-Encoder2
        n_downsampling = 2
        model = []
        i = 0
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer3-Encoder3
        model = []
        i = 1
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_3', nn.Sequential(*model))
        ####################################ART Blocks##############################################
        mult = 4
        self.art_1 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_2 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_3 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_4 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_5 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_6 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_7 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_8 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_9 = ART_block(self.config, input_dim, img_size, transformer=None)
        ############################################################################################
        # Layer13-Decoder1
        n_downsampling = 2
        i = 0
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1, output_padding=0,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer14-Decoder2
        i = 1
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1, output_padding=0,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer15-Decoder3
        model = []
        model = [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_dim, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        setattr(self, 'decoder_3', nn.Sequential(*model))

    ############################################################################################

    def forward(self, x):
        # Encoder
        # x = self.encoder_1(x)
        # x = self.encoder_2(x)
        # x = self.encoder_3(x)

        # print('\033[1;31m---residual_transformers.py---\033[0m')
        # print("x.shape = ", x.shape)
        # x.shape =  torch.Size([12, 256, 64, 64])

        # Information bottleneck
        x = self.art_1(x)
        x = self.art_2(x)
        x = self.art_3(x)
        x = self.art_4(x)
        x = self.art_5(x)
        x = self.art_6(x)
        x = self.art_7(x)
        x = self.art_8(x)
        x = self.art_9(x)
        # Decoder
        # print('\033[1;31m---residual_transformers.py---\033[0m')
        # print("x.shape = ", x.shape)
        x = self.decoder_1(x)
        # print('\033[1;31m---residual_transformers.py---\033[0m')
        # print("x.shape = ", x.shape)
        x = self.decoder_2(x)
        # print('\033[1;31m---residual_transformers.py---\033[0m')
        # print("x.shape = ", x.shape)
        x = self.decoder_3(x)
        # print('\033[1;31m---residual_transformers.py---\033[0m')
        # print("x.shape = ", x.shape)
        return x

class channel_compression(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(channel_compression, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))
        else:
          self.skip = None

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.block(x)
        out += (x if self.skip is None else self.skip(x))
        out = F.relu(out)
        return out
CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-L_16': configs.get_l16_config(),
    'Res-ViT-B_16': configs.get_resvit_b16_config(),
    'Res-ViT-L_16': configs.get_resvit_l16_config(),
}

class DenseNet(nn.Module):
    def __init__(self, n_channels=200, G0=200, kSize=3, D=3, C=4, G=200, dilateSet=[1,2,4,4],output_dim=1):
        super(DenseNet, self).__init__()
        ngf=50
        norm_layer = nn.BatchNorm2d
        padding_type = 'reflect'
        mult = 4
        use_bias = False
        self.D = D   # number of RDB
        self.C = C   # number of Conv in RDB
        self.kSize = kSize  # kernel size
        self.dilateSet = dilateSet   # dilation setting in SERDRB

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(n_channels, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.SEDRDBs = nn.ModuleList()
        for i in range(self.D):
            self.SEDRDBs.append(
                SEDRDB(growRate0=G0, growRate=G, nConvLayers=C, dilateSet=dilateSet)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        # Up-sampling net
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(G0, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(G, 1, kSize, padding=(kSize - 1) // 2, stride=1)
        ])
         # Layer13-Decoder1
        n_downsampling = 2
        i = 0
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1, output_padding=0,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer14-Decoder2
        i = 1
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1, output_padding=0,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer15-Decoder3
        model = []
        model = [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_dim, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        setattr(self, 'decoder_3', nn.Sequential(*model))
        # 信息融合模块
        self.TAGNet1 = nn.Conv2d(2, 32, kSize, padding=(kSize - 1) // 2, stride=1)
        self.REFNet1 = nn.Conv2d(2, 32, kSize, padding=(kSize - 1) // 2, stride=1)
        self.swin_block = Swin_multi_modal_block(img_size=256, patch_size=1, in_chans=200, embed_dim=200)
        # self.swim_block=SwinIR(img_size=256, patch_size=32, in_chans=32, embed_dim=32)
        ############################################################################################
        # 3D卷积相关
        # 输入数据，（N，C，D，H，W），N是batch size,C是通道数，D是深度，H是高度，W是宽度
        # 三维卷积核说明，（FD，FH，FW），FD表示深度方向的滤波器尺寸，FH表示高度，FW表示宽度

        self.conv3D1 = nn.Conv3d(1, 64, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

        self.conv3D_first = nn.Sequential(nn.Conv3d(in_channels = 2,
                                                   out_channels = 32,
                                                   kernel_size = (2, 3, 3),
                                                   stride = (1, 1, 1),
                                                   padding = (0,1,1)
                                                   )
                                        #  nn.BatchNorm3d(32),
                                        #  nn.ReLU())
                                         )

        self.conv3D_second = nn.Sequential(nn.Conv3d(in_channels = 32,
                                                   out_channels = 32,
                                                   kernel_size = (2, 3, 3),
                                                   stride = (1, 1, 1),
                                                   padding = (0,1,1)
                                                   )
                                        #  nn.BatchNorm3d(32),
                                        #  nn.ReLU())
                                         )


        # self.TAGNet2 = nn.Conv2d(64, 32, kSize, padding=(kSize - 1) // 2, stride=1)

#    def forward(self, x):
    def forward(self, x):
        # x_tag_seq2 = torch.unsqueeze(x_tag_past, 2)
        # x_tag_seq1 = torch.unsqueeze(x_tag, 2)
        # x_tag_seq0 = torch.unsqueeze(x_tag_next, 2)
        # x_ref_seq2 = torch.unsqueeze(x_ref_past, 2)
        # x_ref_seq1 = torch.unsqueeze(x_ref, 2)             
        # x_ref_seq0 = torch.unsqueeze(x_ref_next, 2)
        # x_tag_all = torch.cat((x_tag_seq2,x_tag_seq1,x_tag_seq0),2)
        # x_ref_all = torch.cat((x_ref_seq2,x_ref_seq1,x_ref_seq0),2)
        # x_3d_tag_out1 = self.conv3D_first(x_tag_all)
        # x_3d_tag_out2 = self.conv3D_second(x_3d_tag_out1)
        # x_3d_tag_out2_squeeze = torch.squeeze(x_3d_tag_out2,2)
        # x_3d_ref_out1 = self.conv3D_first(x_ref_all)
        # x_3d_ref_out2 = self.conv3D_second(x_3d_ref_out1)
        # x_3d_ref_out2_squeeze = torch.squeeze(x_3d_ref_out2,2)
        # transformer_out_tag, transformer_out_ref = self.swin_block(x_3d_tag_out2_squeeze, x_3d_ref_out2_squeeze)
        # x = transformer_out_tag

#        x = torch.cat((x_tag, x_ref), 1)

        # print('\033[1;31m---networks.py---\033[0m')
        # print("x.shape = ", x.shape)
        # x.shape =  torch.Size([1, 32, 256, 256])



#        f1 = self.SFENet1(x)
#        x = self.SFENet2(f1)

        # print('\033[1;31m---networks.py---\033[0m')
        # print("f1.shape = ", f1.shape)
        # print("x.shape = ", x.shape)
        # f1.shape = torch.Size([1, 32, 256, 256])
        # x.shape = torch.Size([1, 32, 256, 256])
        x=self.swin_block(x)

        # D的值为3
        SEDRDBs_out = []
        for j in range(self.D):
            x = self.SEDRDBs[j](x)
            SEDRDBs_out.append(x)
        


        # print('\033[1;31m---networks.py---\033[0m')
        # print("x.shape = ", x.shape)
        

        x = self.GFF(torch.cat(SEDRDBs_out, 1))

        # print('\033[1;31m---networks.py---\033[0m')
        # print("x.shape = ", x.shape)
        # x.shape =  torch.Size([1, 32, 256, 256])

#        x += f1

        x = self.UPNet(x)
        #  # print('\033[1;31m---residual_transformers.py---\033[0m')
        # # print("x.shape = ", x.shape)
        # x = self.decoder_1(x)
        # # print('\033[1;31m---residual_transformers.py---\033[0m')
        # # print("x.shape = ", x.shape)
        # x = self.decoder_2(x)
        # # print('\033[1;31m---residual_transformers.py---\033[0m')
        # # print("x.shape = ", x.shape)
        # x = self.decoder_3(x)
        # # print('\033[1;31m---networks.py---\033[0m')
        # # print("x.shape = ", x.shape)
        # # print("output.shape = ", output.shape)

        return x


# SEDRDB(growRate0=32, growRate=32, nConvLayers=4, dilateSet=[1,2,4,4])
# Squeeze&Excite Dilated Residual dense block (SEDRDB) architecture
class SEDRDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, dilateSet, kSize=3):
        super(SEDRDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(SEDRDB_Conv(G0 + c * G, G, dilateSet[c], kSize))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

        # Squeeze and Excitation Layer
        self.SE = SELayer(channel=G0, reduction=16)

    def forward(self, x):
        x1 = self.LFF(self.convs(x))
        # print(x1.shape)
        x2 = self.SE(x1)
        # print(x2.shape)
        x3 = x2 + x
        return x3


class SEDRDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, dilate, kSize=3):
        super(SEDRDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=dilate * (kSize - 1) // 2, dilation=dilate, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # torch.Size([1, 32])
        y = self.fc(y).view(b, c, 1, 1)
        # torch.Size([1, 32, 1, 1])
        return x * y.expand_as(x)
    
class Unet_struc(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        # if type(norm_layer) == functools.partial:
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['b'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*4, self.ngf*8, kernel_size=3, padding=1, bias=False),
            norm_layer(self.ngf*8),
            nn.ReLU(True),
            nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=3, padding=1, bias=False),
            norm_layer(self.ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, kernel_size=2, stride=2)
            # norm_layer(self.ngf*2),
            # nn.ReLU(True)
            ]
        self.b = nn.Sequential(*model['b'])

        model['down3'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*2, self.ngf*4, kernel_size=3, padding=1, bias=False),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            nn.Conv2d(self.ngf*4, self.ngf*4, kernel_size=3, padding=1, bias=False),
            norm_layer(self.ngf*4),
            nn.ReLU(True)
            ]
        self.down3 = nn.Sequential(*model['down3'])

        model['down2'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=3, padding=1, bias=False),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            nn.Conv2d(self.ngf*2, self.ngf*2, kernel_size=3, padding=1, bias=False),
            norm_layer(self.ngf*2),
            nn.ReLU(True)
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(self.ngf),
            nn.ReLU(True)
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['up2'] = [
            nn.Conv2d(self.ngf*8, self.ngf*4, kernel_size=3, padding=1, bias=False),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            nn.Conv2d(self.ngf*4, self.ngf*4, kernel_size=3, padding=1, bias=False),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])

        model['up3'] = [
            nn.Conv2d(self.ngf*4, self.ngf*2, kernel_size=3, padding=1, bias=False),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            nn.Conv2d(self.ngf*2, self.ngf*2, kernel_size=3, padding=1, bias=False),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up3 = nn.Sequential(*model['up3'])
        
        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
            # norm_layer(self.output_nc),
            # nn.ReLU(True)
            ]
        self.up1 = nn.Sequential(*model['up1'])
        
        # self.model = model

    def forward(self, input):
        f1 = self.down1(input)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        print(f3.shape)
        print(self.b(f3).shape)
        f4 = torch.cat([f3, self.b(f3)], 1)
        f5 = torch.cat([f2, self.up2(f4)], 1)
        f6 = torch.cat([f1, self.up3(f5)], 1)
        return self.up1(f6)

class Unet(nn.Module):
    def __init__(self, input_nc=200, output_nc=2, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet, self).__init__()
        self.gpu_ids = gpu_ids
        self.fcn = nn.Sequential(nn.Conv2d(input_nc,46,1,1,0),
       			 	norm_layer(46),
       			 	nn.ReLU(True),
       			 	nn.Conv2d(46,46,1,1,0),
       			 	norm_layer(46),
       			 	nn.ReLU(True),
       			 	nn.Conv2d(46,46,1,1,0),
       			 	norm_layer(46),
       			 	nn.ReLU(True),
       			 	nn.Conv2d(46,46,1,1,0),
       			 	norm_layer(46),
       			 	nn.ReLU(True)
                                         )
        self.model = Unet_struc(46, output_nc, ngf, norm_layer, use_dropout, gpu_ids)

    def forward(self, input):
        x=self.fcn(input)
        y=self.model(x)
        return y
