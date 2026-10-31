import ml_collections


### wyz
def get_swin_16_config():
    """Returns the Swin/16 configuration."""
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.transformer.upscale = 1
    config.transformer.window_size = 8 
    config.transformer.depths = 1.0
    config.transformer.depths = 6 #[6, 6, 6, 6, 6, 6]
    config.transformer.embed_dim = 180
    config.transformer.num_heads = 6 #[6, 6, 6, 6, 6, 6]
    config.transformer.mlp_ratio = 2.0
    config.transformer.upsampler = ''

    '''
    model = SwinIR(upscale=1,
                   in_chans=1,
                   img_size=[256, 256],
                   window_size=8,
                   img_range=1.0,
                   depths=[6, 6, 6, 6, 6, 6],
                   embed_dim=180,
                   num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2.0,
                   upsampler='',
    '''

    #config.hidden_size = 768
    #config.transformer = ml_collections.ConfigDict()
    #config.transformer.mlp_dim = 3072
    #config.transformer.num_heads = 12
    #config.transformer.num_layers = 12
    #config.transformer.attention_dropout_rate = 0.0
    #config.transformer.dropout_rate = 0.1
    #config.pretrained_path = './model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    #config.patch_size = 16
    #config.activation = 'softmax'
    #config.patches.grid = (16, 16)
    #config.name = 'b16'
    #config.pretrained_path = './model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'

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
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1


    config.pretrained_path = './model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.activation = 'softmax'
    return config


def get_resvit_b16_config():
    """Returns the residual ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.name = 'b16'

    config.pretrained_path = './model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'

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
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1


    config.pretrained_path = './model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
    return config


def get_resvit_l16_config():
    """Returns the residual ViT-L/16 configuration. customized """
    config = get_l16_config()
    config.patches.grid = (16, 16)
    
    config.name = 'l16'
    config.pretrained_path = './model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
    return config


