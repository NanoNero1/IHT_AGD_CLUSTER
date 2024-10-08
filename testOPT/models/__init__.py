from .mnist import *
from .resnet import *
from .vgg import *
from .linear import *
from .mnist2 import *
from .wideresnet import *

MODELS_MAP = {
    'MNIST2': MNIST2,
    'MNIST2_LEAKY': MNIST2_LEAKY,
    'Linear': LinearRegression,
    'MNIST': MNIST,
    'VGG11': VGG11,
    'VGG13': VGG13,
    'VGG16': VGG16,
    'VGG19': VGG19,
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152,
    'WideResNet2810': WideResNet2810
}