from .mobilenetv2 import mobilenetv2
from .mobilenetv3 import mobilenet_v3_small
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .vgg import vgg11, vgg13, vgg16, vgg11_bn, vgg13_bn, vgg16_bn
from .vision_transformer import vit_b_16
from .swin_transformer import swin_transformer_base, swin_transformer_small, swin_transformer_tiny
from .shufflenet_v2 import shufflenet_v2_x1_0,shufflenet_v2_x0_5,shufflenet_v2_x1_5,shufflenet_v2_x2_0
get_model_from_name = {
    "mobilenetv2"               : mobilenetv2,
    "mobilenet_v3_small"        : mobilenet_v3_small,
    "shufflenet_v2_x1_0"        : shufflenet_v2_x1_0,
    "resnet18"                  : resnet18,
    "resnet34"                  : resnet34,
    "resnet50"                  : resnet50,
    "resnet101"                 : resnet101,
    "resnet152"                 : resnet152,
    "vgg11"                     : vgg11,
    "vgg13"                     : vgg13,
    "vgg16"                     : vgg16,
    "vgg11_bn"                  : vgg11_bn,
    "vgg13_bn"                  : vgg13_bn,
    "vgg16_bn"                  : vgg16_bn,
    "vit_b_16"                  : vit_b_16,
    "swin_transformer_tiny"     : swin_transformer_tiny,
    "swin_transformer_small"    : swin_transformer_small,
    "swin_transformer_base"     : swin_transformer_base
}