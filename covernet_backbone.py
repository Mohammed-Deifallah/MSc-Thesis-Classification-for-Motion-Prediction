from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet50, mobilenet_v2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Backbone(nn.Module):

    def __init__(self, input_shape: Tuple[int, int, int], weights_path: str=None, pretrained: bool=False):
        
        super().__init__()
        self._backbone = None
        self._input_shape = input_shape
        self._weights_path = weights_path
        self._pretrained = pretrained

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        pass
    
    def calculate_backbone_feature_dim(self) -> int:
        ones = torch.ones(1, *self._input_shape)
        output_feat = self(ones)
        return output_feat.shape[-1]

    def _modify_input_channels(self):
        pass
    
    def _load_weights(self):
        pretrained_dict = torch.load(self._weights_path, map_location=device)
        model_dict = self._backbone.state_dict()
        model_dict.update(pretrained_dict)
        self._backbone.load_state_dict(model_dict)


class ResNetBackbone(Backbone):

    def __init__(self, input_shape: Tuple[int, int, int], weights_path: str=None, pretrained: bool=False):
        """
        Inits ResNetBackbone with ResNet-50 
        """
        super().__init__(input_shape, weights_path, pretrained)
        self._backbone = resnet50(pretrained=self._pretrained)
        self._modify_input_channels()
        
        if self._weights_path is not None:
            self._load_weights()
            
        self._backbone = nn.Sequential(*list(self._backbone.children())[:-1])
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Outputs features after last convolution.
        :param input_tensor:  Shape [batch_size, n_channels, length, width].
        :return: Tensor of shape [batch_size, n_convolution_filters]. For resnet50,
            the shape is [batch_size, 2048].
        """
        
        backbone_features = self._backbone(input_tensor)
        return torch.flatten(backbone_features, start_dim=1)
    
    def calculate_backbone_feature_dim(self) -> int:
        return 2048

    def _modify_input_channels(self):
        original_layer = self._backbone.conv1
        self._backbone.conv1 = nn.Conv2d(self._input_shape[0], original_layer.out_channels,
                                        kernel_size=original_layer.kernel_size, 
                                        stride=original_layer.stride,
                                        padding=original_layer.padding,
                                        bias=original_layer.bias)

class MobileNetBackbone(Backbone):

    def __init__(self, input_shape: Tuple[int, int, int], weights_path: str=None, pretrained: bool=False):
        """
        Inits MobileNetBackbone with MobileNet-V2
        """
        super().__init__(input_shape, weights_path, pretrained)
        self._backbone = mobilenet_v2(pretrained=self._pretrained)
        self._modify_input_channels()
        
        if self._weights_path is not None:
            self._load_weights()            
        
        self._backbone = nn.Sequential(*list(self._backbone.children())[:-1])
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Outputs features after last convolution.
        :param input_tensor:  Shape [batch_size, n_channels, length, width].
        :return: Tensor of shape [batch_size, n_convolution_filters]. For mobilenet_v2,
            the shape is [batch_size, 1280].
        """
        backbone_features = self._backbone(input_tensor)
        return backbone_features.mean([2, 3])
    
    def calculate_backbone_feature_dim(self) -> int:
         return 1280

    def _modify_input_channels(self) -> None:
        original_layer = self._backbone.features._modules['0']._modules['0']
        self._backbone.features._modules['0']._modules['0'] = nn.Conv2d(self._input_shape[0], original_layer.out_channels,
                                                                        kernel_size=original_layer.kernel_size,
                                                                        stride=original_layer.stride,
                                                                        padding=original_layer.padding,
                                                                        bias=original_layer.bias)
        