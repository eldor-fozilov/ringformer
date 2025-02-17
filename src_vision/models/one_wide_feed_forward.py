import copy
import logging
import math
import re

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import models.configs as configs

logger = logging.getLogger(__name__)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class PositionalEncoding(nn.Module):
    def __init__(self, n_patches, hidden_dim):
        super(PositionalEncoding, self).__init__()
        self.n_patches = n_patches
        self.hidden_dim = hidden_dim
        #self.device = device

        self.pos_idx = torch.arange(0, self.n_patches)
        self.pos_embed = torch.zeros(self.n_patches, self.hidden_dim)
        
        for i in range(0, self.hidden_dim, 2):
            self.pos_embed[:, i] = np.sin(self.pos_idx / (10000**(i / self.hidden_dim)))
            self.pos_embed[:, i+1] = np.cos(self.pos_idx / (10000**(i / self.hidden_dim)))         
        
        self.pos_embed = nn.Parameter(self.pos_embed.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        assert x.size(1) == self.n_patches
        return self.pos_embed
        
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            # self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
            #                              width_factor=config.resnet.width_factor)
            # in_channels = self.hybrid_model.width * 16
            return NotImplementedError("Hybrid model not implemented yet")
        
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.position_embeddings = PositionalEncoding(n_patches + 1, config.hidden_size) # num_patches + 1 for the cls token

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings(x)
        embeddings = self.dropout(embeddings)
        return embeddings

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.attn_vis = config.attn_vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

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

    def forward(self, Q, K, V):
        mixed_query_layer = self.query(Q)
        mixed_key_layer = self.key(K)
        mixed_value_layer = self.value(V)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.attn_vis else None
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
        self.act_fn = ACT2FN["gelu"]
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

class Att_Block(nn.Module):
    def __init__(self, config):
        super(Att_Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(Q=x, K=x, V=x)
        x = x + h
        return x, weights

class FF_Block(nn.Module):
    def __init__(self, config):
        super(FF_Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.ff_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)

    def forward(self, x):
        h = x
        x = self.ff_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.attn_vis = config.attn_vis
        self.img_size = config.img_size
        self.hidden_dim = config.hidden_size
        self.n_patches = self.img_size // config.patches["size"][0] * self.img_size // config.patches["size"][1]
        self.num_layers = config.transformer["num_layers"]
        
        self.attn_blocks = nn.ModuleList([Att_Block(config) for _ in range(self.num_layers)])
        self.ff_block = FF_Block(config)
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        
        output = hidden_states
        
        attn_weights = []
        for attn_block in self.attn_blocks:
            output, weights = attn_block(output)
            output = self.ff_block(output)
            if self.attn_vis:
                attn_weights.append(weights.detach().cpu())
        
        encoded = self.encoder_norm(output)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        config.img_size = img_size
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class OWF(nn.Module):
    def __init__(self, config, img_size=224, num_classes=1000):
        super(OWF, self).__init__()
        self.num_classes = num_classes
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return logits, loss
        else:
            return logits, attn_weights

    def load_from(self, model_dir):
        # Load from a PyTorch state_dict
        state_dict = torch.load(model_dir)
        self.load_state_dict(state_dict)

OWF_CONFIGS = {
    'OWF-S_16': configs.get_s16_config(),
    'OWF-S_32': configs.get_s32_config(),
    'OWF-B_16': configs.get_b16_config(),
    'OWF-B_32': configs.get_b32_config(),
    'OWF-L_16': configs.get_l16_config(),
    'OWF-L_32': configs.get_l32_config(),
    'OWF-H_14': configs.get_h14_config(),
    'OWF-testing': configs.get_testing(),
}
