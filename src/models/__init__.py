from .unet import UNet
from .munet import mUNet
from .catunet import CatUNet
from .dilunet import DilUNet


def build_model(backbone, init_filters, num_stages, 
                upscale_bilinear = None, num_classes = None, 
                background_class = None, layered_input = False, **kwargs):

    num_channels = num_classes if layered_input else 1

    unet_kwargs = dict( in_channels = num_channels, 
                        out_channels = num_channels, 
                        init_filters = init_filters,
                        num_stages = num_stages, )

    backbone = backbone.lower()

    if backbone == 'catunet':
        unet_kwargs.update(dict(num_classes=num_classes, 
                            background_class=background_class))
        model = CatUNet(**unet_kwargs)

    elif backbone == 'dilunet':
        unet_kwargs.update(dict(upscale_bilinear=upscale_bilinear))
        model = DilUNet(**unet_kwargs)

    elif backbone == 'munet':
        model = mUNet(**unet_kwargs)

    else:
        model = UNet(**unet_kwargs)
    
    return model
