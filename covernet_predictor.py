"""This file is inspired by https://github.com/nutonomy/nuscenes-devkit"""
from typing import List, Tuple, Union, Callable

import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CoverNet(nn.Module):
    """ Implementation of CoverNet https://arxiv.org/pdf/1911.10298.pdf """

    def __init__(self, backbone: nn.Module, num_modes: int = 64,
                 n_hidden_layers: List[int] = [4096], asv_dim: int = 3):

        if not isinstance(n_hidden_layers, list):
            raise ValueError(f"Param n_hidden_layers must be a list. Received {type(n_hidden_layers)}")

        super().__init__()

        self.__backbone = backbone

        backbone_feature_dim = self.__backbone.calculate_backbone_feature_dim()
        n_hidden_layers = [backbone_feature_dim + asv_dim] + n_hidden_layers + [num_modes]
        linear_layers = [nn.Linear(in_dim, out_dim)
                         for in_dim, out_dim in zip(n_hidden_layers[:-1], n_hidden_layers[1:])]
        
        self.__head = nn.ModuleList(linear_layers)
                
        self.__relu = nn.ReLU()
        self.__dropout = nn.Dropout()
        self.__bns = nn.ModuleList([nn.BatchNorm1d(num_features=out_dim) for out_dim in n_hidden_layers[1:]])

    def forward(self, image: torch.Tensor, asv: torch.Tensor) -> torch.Tensor:
        """
        :return: Logits for the batch.
        """
       
        backbone_features = self.__backbone(image)
        logits = torch.cat([backbone_features, asv], dim = 1)
        
        for i in range(len(self.__head)):
            linear, bn = self.__head[i], self.__bns[i]
            logits = (self.__relu(bn(linear(logits))))

        return logits
