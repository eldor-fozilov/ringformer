import copy
import logging
import math

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
            #self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
            #                              width_factor=config.resnet.width_factor)
            # in_channels = self.hybrid_model.width * 16
            raise NotImplementedError("Hybrid model not implemented yet.")
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

    def forward(self, Q, Q_lev_embed, K, K_lev_embed, V, V_lev_embed):
        mixed_query_layer = self.query(Q) + Q_lev_embed
        mixed_key_layer = self.key(K) + K_lev_embed
        mixed_value_layer = self.value(V) + V_lev_embed

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


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.n_patches = config.n_patches
        self.hidden_size = config.hidden_size
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x, attn_lev_info, ff_lev_info, attn_norm, ff_norm):
        h = x
        x = attn_norm(x)
        x, weights = self.attn(Q=x, Q_lev_embed = attn_lev_info[0], K=x, K_lev_embed = attn_lev_info[1], V=x, V_lev_embed = attn_lev_info[2])
        x = x + h

        h = x
        x = torch.matmul(ff_lev_info, x.unsqueeze(dim = -1)).squeeze(dim=-1)
        x = x + h
        
        x = ff_norm(x)
        x = self.ffn(x)
        x = x + h

        return x, weights

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.attn_vis = config.attn_vis
        self.img_size = config.img_size
        self.hidden_dim = config.hidden_size
        self.n_patches = self.img_size // config.patches["size"][0] * self.img_size // config.patches["size"][1]
        self.num_layers = config.transformer["num_layers"]
        self.num_levels = config.transformer["num_levels"]
        self.small_mlp_dim = config.transformer["small_mlp_dim"]
        self.level_encoding = config.level_encoding
        
        config.n_patches = self.n_patches
        
        self.layers = nn.ModuleList([Block(config) for _ in range(self.num_layers)])
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

        # layer norms for each level
        self.ATTN_layer_norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_levels)])
        self.FF_layer_norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_levels)])

        # level signal matrices for attention and FF
        self.ATTN_lev_embed_Q_01 = nn.Parameter(torch.zeros(self.num_levels, config.attn_lev_matrix_rank, self.hidden_dim), requires_grad = True)
        self.ATTN_lev_embed_K_01 = nn.Parameter(torch.zeros(self.num_levels, config.attn_lev_matrix_rank, self.hidden_dim), requires_grad = True)
        self.ATTN_lev_embed_V_01 = nn.Parameter(torch.zeros(self.num_levels, config.attn_lev_matrix_rank, self.hidden_dim), requires_grad = True)
        
        self.ATTN_lev_embed_Q_02 = nn.Parameter(torch.randn(self.num_levels, config.attn_lev_matrix_rank, self.hidden_dim), requires_grad = True)
        self.ATTN_lev_embed_K_02 = nn.Parameter(torch.randn(self.num_levels, config.attn_lev_matrix_rank, self.hidden_dim), requires_grad = True)
        self.ATTN_lev_embed_V_02 = nn.Parameter(torch.randn(self.num_levels, config.attn_lev_matrix_rank, self.hidden_dim), requires_grad = True)
        
        nn.init.normal_(self.ATTN_lev_embed_Q_02, std=1e-6)
        nn.init.normal_(self.ATTN_lev_embed_K_02, std=1e-6)
        nn.init.normal_(self.ATTN_lev_embed_V_02, std=1e-6)
        
        self.lev_embed_FF_01 = nn.Parameter(torch.zeros(self.num_levels, self.small_mlp_dim, self.hidden_dim), requires_grad = True)
        self.lev_embed_FF_02 = nn.Parameter(torch.randn(self.num_levels, self.small_mlp_dim, self.hidden_dim), requires_grad = True)
        
        nn.init.normal_(self.lev_embed_FF_02, std=1e-6)
        
        self.level_indices = torch.arange(1, self.num_levels + 1, requires_grad=False)

    def forward(self, hidden_states):
        
        output = hidden_states
        
        attn_weights = []
        for layer_block in self.layers:
            for level_idx in self.level_indices:
                
                lev_info_Q = self.ATTN_lev_embed_Q_01[level_idx - 1].T @ self.ATTN_lev_embed_Q_02[level_idx - 1]    
                lev_info_K = self.ATTN_lev_embed_K_01[level_idx - 1].T @ self.ATTN_lev_embed_K_02[level_idx - 1]
                lev_info_V = self.ATTN_lev_embed_V_01[level_idx - 1].T @ self.ATTN_lev_embed_V_02[level_idx - 1]

                lev_info_Q = lev_info_Q @ output.unsqueeze(dim = -1)
                lev_info_K = lev_info_K @ output.unsqueeze(dim = -1)
                lev_info_V = lev_info_V @ output.unsqueeze(dim = -1)
            
                ff_lev_info = self.lev_embed_FF_01[level_idx - 1].T @ self.lev_embed_FF_02[level_idx - 1]                
                attn_lev_info = [lev_info_Q.squeeze(dim=-1), lev_info_K.squeeze(dim=-1), lev_info_V.squeeze(dim=-1)]
                
                output, weights = layer_block(output, attn_lev_info, ff_lev_info, self.ATTN_layer_norms[level_idx - 1], self.FF_layer_norms[level_idx - 1])
                
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


class VRingFormer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=1000):
        super(VRingFormer, self).__init__()
        self.attn_vis = config.attn_vis
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
        

VRingFormer_CONFIGS = {
    'VRingFormer-S_16': configs.get_s16_config_vringformer(),
    'VRingFormer-S_32': configs.get_s32_config_vringformer(),
    'VRingFormer-B_16': configs.get_b16_config_vringformer(),
    'VRingFormer-B_32': configs.get_b32_config_vringformer(),
    'VRingFormer-L_16': configs.get_l16_config_vringformer(),
    'VRingFormer-L_32': configs.get_l32_config_vringformer(),
    'VRingFormer-H_14': configs.get_h14_config_vringformer(),
    'VRingFormer-testing': configs.get_testing_vringformer(),
}
