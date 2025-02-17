import ml_collections

def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.num_levels = None
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.attn_vis = True
    config.classifier = 'token'
    return config

# Transformer model configurations.

def get_s16_config():
    """Returns the ViT-S/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 512 # 376 # 392 # 512 # 256
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 2048 # 1024 # 1024 # 2048 # 1024
    config.transformer.num_heads = 8
    config.transformer.num_layers = 6
    config.transformer.num_levels = None
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.attn_vis = True
    config.classifier = 'token'
    return config


def get_s32_config():
    """Returns the ViT-S/32 configuration."""
    config = get_s16_config()
    config.patches.size = (32, 32)
    return config

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.num_levels = None
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.attn_vis = True
    config.classifier = 'token'
    return config


def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    del config.patches.size
    config.patches.grid = (14, 14)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.num_levels = None
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.attn_vis = True
    config.classifier = 'token'
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.num_levels = None
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.attn_vis = True
    config.classifier = 'token'
    return config


# VRingFormer model configurations.

def get_testing_vringformer():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.num_levels = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.attn_vis = True
    config.classifier = 'token'
    config.level_encoding ='learnable'
    return config

def get_s16_config_vringformer():
    """Returns the ViT-S/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 728 # changed from 512 to scale up and make it comparable to bigger models such as OWF
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072 # changed from 2048 to scale up and make it comparable to bigger models such as OWF 
    config.transformer.small_mlp_dim = config.hidden_size // 16
    config.attn_lev_matrix_rank = config.hidden_size // 16
    config.transformer.num_heads = 8
    config.transformer.num_layers = 1
    config.transformer.num_levels = 6
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.attn_vis = True
    config.classifier = 'token'
    config.level_encoding = 'learnable'
    return config


def get_s32_config_vringformer():
    """Returns the ViT-S/32 configuration."""
    config = get_s16_config_vringformer()
    config.patches.size = (32, 32)
    return config


def get_b16_config_vringformer():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1284 # changed from 768 to scale up and make it comparable to bigger models such as OWF
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120 # changed from 3072 to scale up and make it comparable to bigger models such as OWF
    config.transformer.small_mlp_dim = config.hidden_size // 16
    config.attn_lev_matrix_rank = config.hidden_size // 16
    config.transformer.num_heads = 12
    config.transformer.num_layers = 1
    config.transformer.num_levels = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.attn_vis = True
    config.classifier = 'token'
    config.level_encoding = 'learnable'
    return config


def get_b32_config_vringformer():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config_vringformer()
    config.patches.size = (32, 32)
    return config


def get_l16_config_vringformer():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.small_mlp_dim = config.hidden_size // 4
    config.attn_lev_matrix_rank = config.hidden_size // 4
    config.transformer.num_heads = 16
    config.transformer.num_layers = 1
    config.transformer.num_levels = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.attn_vis = True
    config.classifier = 'token'
    config.level_encoding = 'learnable'
    return config


def get_l32_config_vringformer():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config_vringformer()
    config.patches.size = (32, 32)
    return config


def get_h14_config_vringformer():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.small_mlp_dim = config.hidden_size // 2
    config.attn_lev_matrix_rank = config.hidden_size // 2
    config.transformer.num_heads = 16
    config.transformer.num_layers = 1
    config.transformer.num_levels = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.attn_vis = True
    config.classifier = 'token'
    config.level_encoding = 'learnable'
    return config

# UiT model configurations.


def get_testing_uit():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.num_levels = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.attn_vis = True
    config.classifier = 'token'
    config.level_encoding ='static'
    return config

def get_s16_config_uit():
    """Returns the ViT-S/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 848 # changed from 512 to scale up and make it comparable to bigger models such as OWF
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072 # changed from 2048 to scale up and make it comparable to bigger models such as OWF
    config.transformer.small_mlp_dim = config.hidden_size // 16
    config.attn_lev_matrix_rank = config.hidden_size // 16
    config.transformer.num_heads = 8
    config.transformer.num_layers = 1
    config.transformer.num_levels = 6
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.attn_vis = True
    config.classifier = 'token'
    config.level_encoding = 'static'
    return config


def get_s32_config_uit():
    """Returns the ViT-S/32 configuration."""
    config = get_s16_config_vringformer()
    config.patches.size = (32, 32)
    return config


def get_b16_config_uit():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1560 # changed from 768 to scale up and make it comparable to bigger models such as OWF
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 6240 # changed from 3072 to scale up and make it comparable to bigger models such as OWF
    config.transformer.small_mlp_dim = config.hidden_size // 16
    config.attn_lev_matrix_rank = config.hidden_size // 16
    config.transformer.num_heads = 12
    config.transformer.num_layers = 1
    config.transformer.num_levels = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.attn_vis = True
    config.classifier = 'token'
    config.level_encoding = 'static'
    return config


def get_b32_config_uit():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config_vringformer()
    config.patches.size = (32, 32)
    return config


def get_l16_config_uit():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.small_mlp_dim = config.hidden_size // 4
    config.attn_lev_matrix_rank = config.hidden_size // 4
    config.transformer.num_heads = 16
    config.transformer.num_layers = 1
    config.transformer.num_levels = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.attn_vis = True
    config.classifier = 'token'
    config.level_encoding = 'static'
    return config


def get_l32_config_uit():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config_vringformer()
    config.patches.size = (32, 32)
    return config


def get_h14_config_uit():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.small_mlp_dim = config.hidden_size // 2
    config.attn_lev_matrix_rank = config.hidden_size // 2
    config.transformer.num_heads = 16
    config.transformer.num_layers = 1
    config.transformer.num_levels = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.attn_vis = True
    config.classifier = 'token'
    config.level_encoding = 'static'
    return config