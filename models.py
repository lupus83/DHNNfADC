import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from typing import Any, Dict
from collections import OrderedDict
from colliotnet import Conv4_FC3
import logging

LOG = logging.getLogger(__name__)

def conv3d(in_channels, out_channels, kernel_size=3, stride=1):
    if kernel_size != 1:
        padding = 1
    else:
        padding = 0
    return nn.Conv3d(in_channels, out_channels,
                     kernel_size,
                     stride=stride,
                     padding=padding,
                     bias=False)

class ConvBnReLU(nn.Module):

    def __init__(self, in_channels, out_channels, bn_momentum=0.05, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                     kernel_size,
                     stride=stride,
                     padding=padding,
                     bias=False)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, bn_momentum=0.05, stride=1):
        super().__init__()
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DynamicBlock(nn.Module):
    # motivated by
    # https://arxiv.org/abs/1605.09673 and
    # https://papers.nips.cc/paper/2017/hash/e7e23670481ac78b3c4122a99ba60573-Abstract.html
    def __init__(self, in_channels, out_channels, bn_momentum=0.1, ndim_non_img=31, kernel_size=3, stride=1, hidden_dim=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        LOG.debug(f'Stride of Dynamic Conv Layer: {self.stride}')

        self.kernel_shape = [out_channels, in_channels, kernel_size, kernel_size, kernel_size]
        self.n_elements_per_channel = np.prod(self.kernel_shape[1:])
        if kernel_size != 1:
            self.padding = 1
        else:
            self.padding = 0
        self.aux_base = nn.Linear(ndim_non_img, hidden_dim, bias=False)
        self.aux_relu = nn.ReLU()
        self.aux_dropout = nn.Dropout(p=0.2, inplace=True)
        self.aux_filtergenerator = nn.Linear(hidden_dim, np.prod(self.kernel_shape), bias=False)

        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x_aux):

        # 0. get kernel weights from auxiliary network
        kernel = self.aux_base(x_aux)
        kernel = self.aux_relu(kernel)
        kernel = self.aux_dropout(kernel)
        kernel = self.aux_filtergenerator(kernel)  # shape: (batch_size, out_channels*in_channels*kernelize**3)
        batch_size = kernel.size(0)

        assert x.size(0) == batch_size and x.size(1) == self.in_channels
        
        # 1. reorder kernel weights
        kernel = kernel.view(batch_size*self.out_channels, self.n_elements_per_channel)

        # 2. unravel kernels
        kernel = kernel.view(batch_size*self.out_channels, *self.kernel_shape[1:])

        # 3. reorder input x
        out = x.view(batch_size*self.in_channels, *x.size()[2:])
        out = out.unsqueeze(0)

        # 4. calc new feature map
        out = F.conv3d(out, kernel, stride=self.stride, padding=self.padding, groups=batch_size) # set torch.backends.cudnn.deterministic = True for deterministic algo and reproducible results

        # 5. reorder output as (batch_size, out_channels, weight, height, depth)
        out = out.squeeze(0)
        out = out.view(batch_size, self.out_channels, *out.size()[1:])

        # bn relu
        out = self.bn(out)
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, in_channels=1, n_outputs=1, bn_momentum=0.05, n_basefilters=8):
        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)  # input size: 1, 64, 64, 64
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)  # 32
        self.block2 = ResBlock(n_basefilters, 2*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2*n_basefilters, 4*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4*n_basefilters, 8*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8*n_basefilters, n_outputs)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class ConcatHNN(nn.Module):
    # ConcatHNN1FC
    def __init__(self,
        in_channels: int=1,
        n_outputs: int=1,
        bn_momentum: float=0.05,
        n_basefilters: int=32,
        ndim_non_img: int=31):

        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2*n_basefilters, 4*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4*n_basefilters, 8*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8*n_basefilters+ndim_non_img, n_outputs)

    def forward(self, x, non_image_data):

        out = self.conv1(x)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = torch.cat((out, non_image_data), dim=1)
        out = self.fc(out)

        return out

class ConcatHNN_V2(nn.Module):
    # ConcatHNN2FC
    def __init__(self,
        in_channels: int=1,
        n_outputs: int=1,
        bn_momentum: float=0.05,
        n_basefilters: int=32,
        ndim_non_img: int=31):

        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2*n_basefilters, 4*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4*n_basefilters, 8*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        layers = [('fc1', nn.Linear(8*n_basefilters+ndim_non_img, 12)),
                  ('dropout', nn.Dropout(p=0.5, inplace=True)),
                  ('relu', nn.ReLU()),
                  ('fc2', nn.Linear(12, n_outputs))]
        self.fc = nn.Sequential(OrderedDict(layers))

    def forward(self, x, non_image_data):

        out = self.conv1(x)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = torch.cat((out, non_image_data), dim=1)
        out = self.fc(out)

        return out

class DynamicHNN(nn.Module):
    
    def __init__(self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float,
        n_basefilters: int,
        dynamic_conv_args: Dict[Any, Any],
        ):
    
        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2*n_basefilters, 4*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = DynamicBlock(4*n_basefilters, 8*n_basefilters, bn_momentum=bn_momentum, **dynamic_conv_args)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8*n_basefilters, n_outputs)

    def forward(self, x, x_aux):

        out = self.conv1(x)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out, x_aux)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class InteractiveHNN(nn.Module):
    # motivated by https://link.springer.com/chapter/10.1007%2F978-3-030-59713-9_24
    def __init__(self,
        in_channels: int=1,
        n_outputs: int=3,
        bn_momentum: float=0.05,
        n_basefilters: int=8,
        ndim_non_img: int=31
        ):
    
        super().__init__()

        # ResNet
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2*n_basefilters, 4*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4*n_basefilters, 8*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8*n_basefilters, n_outputs)
        
        self.aux_base = nn.Linear(ndim_non_img, 8, bias=False)
        self.aux_relu = nn.ReLU()
        self.aux_dropout = nn.Dropout(p=0.2, inplace=True)
        self.aux_1 = nn.Linear(8, n_basefilters, bias=False)
        self.aux_2 = nn.Linear(n_basefilters, n_basefilters, bias=False)
        self.aux_3 = nn.Linear(n_basefilters, 2*n_basefilters, bias=False)
        self.aux_4 = nn.Linear(2*n_basefilters, 4*n_basefilters, bias=False)

    def forward(self, x, x_aux):

        attention = self.aux_base(x_aux)
        attention = self.aux_relu(attention)
        attention = self.aux_dropout(attention)

        out = self.conv1(x)
        out = self.pool1(out)

        attention = self.aux_1(attention)
        batch_size, n_channels, D, H, W = out.size()
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block1(out)

        attention = self.aux_2(attention)
        batch_size, n_channels, D, H, W = out.size()
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block2(out)

        attention = self.aux_3(attention)
        batch_size, n_channels, D, H, W = out.size()
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block3(out)

        attention = self.aux_4(attention)
        batch_size, n_channels, D, H, W = out.size()
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block4(out)

        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class FilmBlock(nn.Module):
    # motivated by https://arxiv.org/abs/1709.07871
    def __init__(self, in_channels, out_channels, bn_momentum=0.1, stride=2,
        ndim_non_img=31, location=0, activation='linear', scale=True, shift=True):
        super().__init__()
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum)
            )
        else:
            self.downsample = None

        # location decoding
        self.location = location
        film_dims = 0
        if location in [0, 1, 3]:
            film_dims = in_channels
        elif location in [2, 4, 5, 6, 7]:
            film_dims = out_channels
        else:
            raise ValueError(f'Invalid location specified: {location}')

        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = film_dims
            self.scale = None
            self.shift = None
            film_dims = 2*film_dims
        elif scale == False and shift == False:
            raise ValueError(f'FilmBlock must either do scale or shift')
        elif scale == False:
            self.scale = 1
            self.shift = None
        elif shift == False:
            self.shift = 0
            self.scale = None
        else:
            raise ValueError('Unkown error occured')

        # create aux net
        layers = [('aux_base', nn.Linear(ndim_non_img, 8, bias=False)),
                              ('aux_relu', nn.ReLU()),
                              ('aux_dropout', nn.Dropout(p=0.2, inplace=True)),
                              ('aux_out', nn.Linear(8, film_dims, bias=False))]
        self.aux=nn.Sequential(OrderedDict(layers))
        if activation == 'sigmoid':
            self.scale_activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.scale_activation = nn.Tanh()
        elif activation == 'linear':
            self.scale_activation = None
            LOG.debug(f'Linear Activation -> No activation function')
        else:
            raise ValueError(f'Invalid input on activation {activation}')

        # sanity check
        if self.location == 2 and self.downsample is None:
            raise ValueError('this setup is equivalent to location=1 and no downsampling!')
        LOG.debug(f'location: {self.location}')
        LOG.debug(f'scale: {self.scale}')
        LOG.debug(f'shift: {self.shift}')
        LOG.debug(f'split size: {self.split_size}')

    def rescale_features(self, feature_map, x_aux):

        attention = self.aux(x_aux)
        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
            v_scale = v_scale.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
            v_shift = v_shift.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale == None:
            v_scale = attention
            v_scale= v_scale.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift == None:
            v_scale = self.scale
            v_shift = attention
            v_shift = v_shift.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
        else:
            raise Exception(f'Ooops, something went wrong: {self.scale}, {self.shift}')

        return (v_scale * feature_map) + v_shift

    def forward(self, x, x_aux):

        if self.location == 0:
            x = self.rescale_features(x, x_aux)
        
        residual = x

        if self.location == 1:
            residual = self.rescale_features(residual, x_aux)

        if self.location == 3:
            x = self.rescale_features(x, x_aux)
        out = self.conv1(x)
        out = self.bn1(out)

        if self.location == 4:
            out = self.rescale_features(out, x_aux)
        out = self.relu(out)

        if self.location == 5:
            out = self.rescale_features(out, x_aux)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
            if self.location == 2:
                residual = self.rescale_features(residual, x_aux)

        if self.location == 6:
            out = self.rescale_features(out, x_aux)
        out += residual

        if self.location == 7:
            out = self.rescale_features(out, x_aux)
        out = self.relu(out)

        return out

class SEFilmBlock_CAT(nn.Module):
    # Block for ZeCatNet
    def __init__(self, in_channels, out_channels, bn_momentum=0.1, stride=2,
        ndim_non_img=31, location=0, activation='linear', scale=True, shift=True):
        super().__init__()
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum)
            )
        else:
            self.downsample = None

        # location decoding
        self.location = location
        film_dims = 0
        if location in [0, 1, 3]:
            film_dims = in_channels
        elif location in [2, 4, 5, 6, 7]:
            film_dims = out_channels
        else:
            raise ValueError(f'Invalid location specified: {location}')

        aux_input_dims = film_dims
        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = film_dims
            self.scale = None
            self.shift = None
            film_dims = 2*film_dims
        elif scale == False and shift == False:
            raise ValueError(f'FilmBlock must either do scale or shift')
        elif scale == False:
            self.scale = 1
            self.shift = None
        elif shift == False:
            self.shift = 0
            self.scale = None
        else:
            raise ValueError('Unkown error occured')

        # create aux net
        layers = [('aux_base', nn.Linear(ndim_non_img+aux_input_dims, 8, bias=False)),
                              ('aux_relu', nn.ReLU()),
                              ('aux_dropout', nn.Dropout(p=0.2, inplace=True)),
                              ('aux_out', nn.Linear(8, film_dims, bias=False))]
        self.aux=nn.Sequential(OrderedDict(layers))
        if activation == 'sigmoid':
            self.scale_activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.scale_activation = nn.Tanh()
        elif activation == 'linear':
            self.scale_activation = None
            LOG.debug(f'Linear Activation -> No activation function')
        else:
            raise ValueError(f'Invalid input on activation {activation}')

        # sanity check
        if self.location == 2 and self.downsample is None:
            raise ValueError('this setup is equivalent to location=1 and no downsampling!')
        LOG.debug(f'location: {self.location}')
        LOG.debug(f'scale: {self.scale}')
        LOG.debug(f'shift: {self.shift}')
        LOG.debug(f'split size: {self.split_size}')


    def rescale_features(self, feature_map, x_aux):

        squeeze = self.global_pool(feature_map)
        squeeze = squeeze.view(squeeze.size(0), -1)
        squeeze = torch.cat((squeeze, x_aux), dim=1)

        attention = self.aux(squeeze)
        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
            v_scale = v_scale.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
            v_shift = v_shift.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale == None:
            v_scale = attention
            v_scale= v_scale.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift == None:
            v_scale = self.scale
            v_shift = attention
            v_shift = v_shift.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
        else:
            raise Exception(f'Ooops, something went wrong: {self.scale}, {self.shift}')

        return (v_scale * feature_map) + v_shift

    def forward(self, x, x_aux):
        
        if self.location == 0:
            x = self.rescale_features(x, x_aux)
        
        residual = x

        if self.location == 1:
            residual = self.rescale_features(residual, x_aux)

        if self.location == 3:
            x = self.rescale_features(x, x_aux)
        out = self.conv1(x)
        out = self.bn1(out)

        if self.location == 4:
            out = self.rescale_features(out, x_aux)
        out = self.relu(out)

        if self.location == 5:
            out = self.rescale_features(out, x_aux)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
            if self.location == 2:
                residual = self.rescale_features(residual, x_aux)

        if self.location == 6:
            out = self.rescale_features(out, x_aux)
        out += residual

        if self.location == 7:
            out = self.rescale_features(out, x_aux)
        out = self.relu(out)

        return out

class SEFilmBlock_FIB(nn.Module):
    # Block for ZeFIBNet
    def __init__(self, in_channels, out_channels, bn_momentum=0.1, stride=2,
        ndim_non_img=31, location=0, activation='linear', scale=True, shift=True):
        super().__init__()
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum)
            )
        else:
            self.downsample = None

        # location decoding
        self.location = location
        film_dims = 0
        if location in [0, 1, 3]:
            film_dims = in_channels
        elif location in [2, 4, 5, 6, 7]:
            film_dims = out_channels
        else:
            raise ValueError(f'Invalid location specified: {location}')

        squeeze_in = film_dims
        film_dims = 8
        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = film_dims
            self.scale = None
            self.shift = None
            film_dims = 2*film_dims
        elif scale == False and shift == False:
            raise ValueError(f'FilmBlock must either do scale or shift')
        elif scale == False:
            self.scale = 1
            self.shift = None
        elif shift == False:
            self.shift = 0
            self.scale = None
        else:
            raise ValueError('Unkown error occured')

        # create aux net
        layers = [('aux_base', nn.Linear(ndim_non_img, 8, bias=False)),
                              ('aux_relu', nn.ReLU()),
                              ('aux_dropout', nn.Dropout(p=0.2, inplace=True)),
                              ('aux_bottleneck', nn.Linear(8, film_dims, bias=False))]
        self.aux = nn.Sequential(OrderedDict(layers))
        self.squeeze = nn.Linear(squeeze_in, 8, bias=False)
        layers2 = [('aux_out_relu', nn.ReLU()),
                        ('aux_out_dropout', nn.Dropout(p=0.2, inplace=True)),
                        ('aux_out', nn.Linear(8, squeeze_in, bias=False)),
                        ('excite_activation', nn.Sigmoid())]
        self.aux_out = nn.Sequential(OrderedDict(layers2))

        if activation == 'sigmoid':
            self.scale_activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.scale_activation = nn.Tanh()
        elif activation == 'linear':
            self.scale_activation = None
            LOG.debug(f'Linear Activation -> No activation function')
        else:
            raise ValueError(f'Invalid input on activation {activation}')

        # sanity check
        if self.location == 2 and self.downsample is None:
            raise ValueError('this setup is equivalent to location=1 and no downsampling!')
        LOG.debug(f'location: {self.location}')
        LOG.debug(f'scale: {self.scale}')
        LOG.debug(f'shift: {self.shift}')
        LOG.debug(f'split size: {self.split_size}')

    def rescale_features(self, feature_map, x_aux):

        se_vector = self.global_pool(feature_map)
        se_vector = se_vector.view(se_vector.size(0), -1)
        se_vector = self.squeeze(se_vector)
        
        attention = self.aux(x_aux)

        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
            assert v_scale.size() == se_vector.size() and v_shift.size() == se_vector.size()
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale == None:
            v_scale = attention
            assert v_scale.size() == se_vector.size()
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift == None:
            v_scale = self.scale
            v_shift = attention
            v_shift.size() == se_vector.size()
        else:
            raise Exception(f'Ooops, something went wrong: {self.scale}, {self.shift}')
        
        se_vector = v_scale * se_vector + v_shift
        se_vector = self.aux_out(se_vector)
        se_vector = se_vector.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)

        return se_vector * feature_map

    def forward(self, x, x_aux):
        
        if self.location == 0:
            x = self.rescale_features(x, x_aux)
        
        residual = x

        if self.location == 1:
            residual = self.rescale_features(residual, x_aux)

        if self.location == 3:
            x = self.rescale_features(x, x_aux)
        out = self.conv1(x)
        out = self.bn1(out)

        if self.location == 4:
            out = self.rescale_features(out, x_aux)
        out = self.relu(out)

        if self.location == 5:
            out = self.rescale_features(out, x_aux)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
            if self.location == 2:
                residual = self.rescale_features(residual, x_aux)

        if self.location == 6:
            out = self.rescale_features(out, x_aux)
        out += residual

        if self.location == 7:
            out = self.rescale_features(out, x_aux)
        out = self.relu(out)

        return out

class SEFilmBlock_NULL(nn.Module):
    # Block for ZeNullNet
    def __init__(self, in_channels, out_channels, bn_momentum=0.1, stride=2,
        ndim_non_img=31, location=0, activation='linear', scale=True, shift=True):
        super().__init__()
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum)
            )
        else:
            self.downsample = None

        # location decoding
        self.location = location
        film_dims = 0
        if location in [0, 1, 3]:
            film_dims = in_channels
        elif location in [2, 4, 5, 6, 7]:
            film_dims = out_channels
        else:
            raise ValueError(f'Invalid location specified: {location}')

        squeeze_dims = film_dims
        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = film_dims
            self.scale = None
            self.shift = None
            film_dims = 2*film_dims
        elif scale == False and shift == False:
            raise ValueError(f'FilmBlock must either do scale or shift')
        elif scale == False:
            self.scale = 1
            self.shift = None
        elif shift == False:
            self.shift = 0
            self.scale = None
        else:
            raise ValueError('Unkown error occured')

        # create aux net
        layers = [('aux_base', nn.Linear(ndim_non_img, 8)),
                  ('aux_relu', nn.ReLU()),
                  ('aux_dropout', nn.Dropout(p=0.2, inplace=True)),
                  ('aux_out', nn.Linear(8, 8*film_dims, bias=False))]
        self.aux = nn.Sequential(OrderedDict(layers))
        self.squeeze = nn.Linear(squeeze_dims, 8, bias=False)

        if activation == 'sigmoid':
            self.scale_activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.scale_activation = nn.Tanh()
        elif activation == 'linear':
            self.scale_activation = None
            LOG.debug(f'Linear Activation -> No activation function')
        else:
            raise ValueError(f'Invalid input on activation {activation}')

        # sanity check
        if self.location == 2 and self.downsample is None:
            raise ValueError('this setup is equivalent to location=1 and no downsampling!')
        LOG.debug(f'location: {self.location}')
        LOG.debug(f'scale: {self.scale}')
        LOG.debug(f'shift: {self.shift}')
        LOG.debug(f'split size: {self.split_size}')

    def rescale_features(self, feature_map, x_aux):

        squeeze_vector = self.global_pool(feature_map)
        squeeze_vector = squeeze_vector.view(squeeze_vector.size(0), -1)
        squeeze_vector = self.squeeze(squeeze_vector)

        weights = self.aux(x_aux)  # matrix weights
        weights = weights.view(*squeeze_vector.size(), -1)  # squeeze_vecotr.size is (batch_size, 8). After reshaping, weights has size (batch_size, 8, film_dims)
        weights = torch.einsum('bi,bij->bj', squeeze_vector, weights)  # j = alpha and beta

        if self.scale == self.shift:
            v_scale, v_shift = torch.split(weights, self.split_size, dim=1)
            v_scale = v_scale.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
            v_shift = v_shift.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale == None:
            v_scale = weights
            v_scale= v_scale.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift == None:
            v_scale = self.scale
            v_shift = weights
            v_shift = v_shift.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
        else:
            raise Exception(f'Ooops, something went wrong: {self.scale}, {self.shift}')

        return (v_scale * feature_map) + v_shift

    def forward(self, x, x_aux):
        
        if self.location == 0:
            x = self.rescale_features(x, x_aux)
        
        residual = x

        if self.location == 1:
            residual = self.rescale_features(residual, x_aux)

        if self.location == 3:
            x = self.rescale_features(x, x_aux)
        out = self.conv1(x)
        out = self.bn1(out)

        if self.location == 4:
            out = self.rescale_features(out, x_aux)
        out = self.relu(out)

        if self.location == 5:
            out = self.rescale_features(out, x_aux)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
            if self.location == 2:
                residual = self.rescale_features(residual, x_aux)

        if self.location == 6:
            out = self.rescale_features(out, x_aux)
        out += residual

        if self.location == 7:
            out = self.rescale_features(out, x_aux)
        out = self.relu(out)

        return out

class SEFilmBlock_NU(nn.Module):
    # Block for ZeNuNet
    def __init__(self, in_channels, out_channels, bn_momentum=0.1, stride=2,
        ndim_non_img=31, location=0, activation='linear', scale=True, shift=True):
        super().__init__()
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum)
            )
        else:
            self.downsample = None

        # location decoding
        self.location = location
        film_dims = 0
        if location in [0, 1, 3]:
            film_dims = in_channels
        elif location in [2, 4, 5, 6, 7]:
            film_dims = out_channels
        else:
            raise ValueError(f'Invalid location specified: {location}')

        squeeze_dims = film_dims
        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = film_dims
            self.scale = None
            self.shift = None
            film_dims = 2*film_dims
        elif scale == False and shift == False:
            raise ValueError(f'FilmBlock must either do scale or shift')
        elif scale == False:
            self.scale = 1
            self.shift = None
        elif shift == False:
            self.shift = 0
            self.scale = None
        else:
            raise ValueError('Unkown error occured')

        # create aux net
        layers = [('aux_base', nn.Linear(ndim_non_img, 8)),
                  ('aux_relu', nn.ReLU()),
                  ('aux_dropout', nn.Dropout(p=0.2, inplace=True)),
                  ('aux_out', nn.Linear(8, 8+film_dims, bias=False))]
        self.aux = nn.Sequential(OrderedDict(layers))
        self.squeeze = nn.Linear(squeeze_dims, 8, bias=False)

        if activation == 'sigmoid':
            self.scale_activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.scale_activation = nn.Tanh()
        elif activation == 'linear':
            self.scale_activation = None
            LOG.debug(f'Linear Activation -> No activation function')
        else:
            raise ValueError(f'Invalid input on activation {activation}')

        # sanity check
        if self.location == 2 and self.downsample is None:
            raise ValueError('this setup is equivalent to location=1 and no downsampling!')
        LOG.debug(f'location: {self.location}')
        LOG.debug(f'scale: {self.scale}')
        LOG.debug(f'shift: {self.shift}')
        LOG.debug(f'split size: {self.split_size}')

    def rescale_features(self, feature_map, x_aux):

        squeeze_vector = self.global_pool(feature_map)
        squeeze_vector = squeeze_vector.view(squeeze_vector.size(0), -1)
        squeeze_vector = self.squeeze(squeeze_vector)

        low_rank = self.aux(x_aux)  # matrix weights, shape (batch_size, 8+FilmDims)
        v0, v1 = torch.split(low_rank, [8, low_rank.size(1)-8], dim=1)  # v0 size -> (batchsize, 8), v1 size -> (batchsize, FilmDims)

        weights = torch.einsum('bi, bj->bij', v0, v1)  # weights size -> (batchsize, 8, FilmDims)
        weights = torch.einsum('bi,bij->bj', squeeze_vector, weights)  # j = alpha and beta = filmdims

        if self.scale == self.shift:
            v_scale, v_shift = torch.split(weights, self.split_size, dim=1)
            v_scale = v_scale.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
            v_shift = v_shift.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale == None:
            v_scale = weights
            v_scale= v_scale.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift == None:
            v_scale = self.scale
            v_shift = weights
            v_shift = v_shift.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
        else:
            raise Exception(f'Ooops, something went wrong: {self.scale}, {self.shift}')

        return (v_scale * feature_map) + v_shift

    def forward(self, x, x_aux):
        
        if self.location == 0:
            x = self.rescale_features(x, x_aux)
        
        residual = x

        if self.location == 1:
            residual = self.rescale_features(residual, x_aux)

        if self.location == 3:
            x = self.rescale_features(x, x_aux)
        out = self.conv1(x)
        out = self.bn1(out)

        if self.location == 4:
            out = self.rescale_features(out, x_aux)
        out = self.relu(out)

        if self.location == 5:
            out = self.rescale_features(out, x_aux)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
            if self.location == 2:
                residual = self.rescale_features(residual, x_aux)

        if self.location == 6:
            out = self.rescale_features(out, x_aux)
        out += residual

        if self.location == 7:
            out = self.rescale_features(out, x_aux)
        out = self.relu(out)

        return out


class ZeCatNet(nn.Module):
    # 
    def __init__(self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float,
        n_basefilters: int,
        filmblock_args: Dict[Any, Any]
        ):
    
        super().__init__()

        self.split_size = 4*n_basefilters
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2*n_basefilters, 4*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.blockX = SEFilmBlock_CAT(4*n_basefilters, 8*n_basefilters, bn_momentum=bn_momentum, **filmblock_args)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8*n_basefilters, n_outputs)

    def forward(self, x, x_aux):

        out = self.conv1(x)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.blockX(out, x_aux)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class ZeFIBNet(nn.Module):
    # 
    def __init__(self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float,
        n_basefilters: int,
        filmblock_args: Dict[Any, Any]
        ):
    
        super().__init__()

        self.split_size = 4*n_basefilters
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2*n_basefilters, 4*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.blockX = SEFilmBlock_FIB(4*n_basefilters, 8*n_basefilters, bn_momentum=bn_momentum, **filmblock_args)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8*n_basefilters, n_outputs)

    def forward(self, x, x_aux):

        out = self.conv1(x)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.blockX(out, x_aux)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class ZeNullNet(nn.Module):
    # 
    def __init__(self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float,
        n_basefilters: int,
        filmblock_args: Dict[Any, Any]
        ):
    
        super().__init__()

        self.split_size = 4*n_basefilters
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2*n_basefilters, 4*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.blockX = SEFilmBlock_NULL(4*n_basefilters, 8*n_basefilters, bn_momentum=bn_momentum, **filmblock_args)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8*n_basefilters, n_outputs)

    def forward(self, x, x_aux):

        out = self.conv1(x)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.blockX(out, x_aux)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class ZeNuNet(nn.Module):
    # 
    def __init__(self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float,
        n_basefilters: int,
        filmblock_args: Dict[Any, Any]
        ):
    
        super().__init__()

        self.split_size = 4*n_basefilters
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2*n_basefilters, 4*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.blockX = SEFilmBlock_NU(4*n_basefilters, 8*n_basefilters, bn_momentum=bn_momentum, **filmblock_args)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8*n_basefilters, n_outputs)

    def forward(self, x, x_aux):

        out = self.conv1(x)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.blockX(out, x_aux)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class SENet(nn.Module):

    def __init__(self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float,
        n_basefilters: int
        ):
    
        super().__init__()

        self.split_size = 4*n_basefilters
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2*n_basefilters, 4*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4*n_basefilters, 8*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8*n_basefilters, n_outputs)

        # create aux net
        layers = [('aux_base', nn.Linear(4*n_basefilters, 8, bias=False)),
                  ('aux_relu', nn.ReLU()),
                  ('aux_out', nn.Linear(8, 4*n_basefilters, bias=False)),
                  ('activation', nn.Sigmoid())]
        self.aux=nn.Sequential(OrderedDict(layers))

    def forward(self, x):

        out = self.conv1(x)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        squeeze = self.global_pool(out)
        squeeze = squeeze.view(squeeze.size(0), -1)
        squeeze = self.aux(squeeze)
        squeeze = squeeze.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(out)

        out = squeeze * out

        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class FilmHNN(nn.Module):

    def __init__(self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float,
        n_basefilters: int,
        filmblock_args: Dict[Any, Any]
        ):
    
        super().__init__()

        self.split_size = 4*n_basefilters
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2*n_basefilters, 4*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.blockX = FilmBlock(4*n_basefilters, 8*n_basefilters, bn_momentum=bn_momentum, **filmblock_args)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8*n_basefilters, n_outputs)

    def forward(self, x, x_aux):

        out = self.conv1(x)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.blockX(out, x_aux)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class CheckAux(nn.Module):

    def __init__(self, in_channels, n_outputs, ndim_non_img, n_hidden_dims, activation, n_features, scale, shift, bn_momentum):

        super().__init__()
        # shift and scale decoding
        
        self.fc = nn.Linear(n_features, 1)
        self.split_size = 0
        if scale and shift:
            self.split_size = n_features
            self.scale = None
            self.shift = None
            n_features = 2*n_features
        elif scale == False and shift == False:
            raise ValueError(f'FilmBlock must either do scale or shift')
        elif scale == False:
            self.scale = 1
            self.shift = None
        elif shift == False:
            self.shift = 0
            self.scale = None
        else:
            raise ValueError('Unkown error occured')
        layers = [('aux_base', nn.Linear(ndim_non_img, n_hidden_dims, bias=False)),
                  ('aux_relu', nn.ReLU()),
                  ('aux_dropout', nn.Dropout(p=0.2, inplace=True)),
                  ('aux_out', nn.Linear(n_hidden_dims, n_features, bias=False))]
        self.aux=nn.Sequential(OrderedDict(layers))
        if activation == 'sigmoid':
            self.scale_activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.scale_activation = nn.Tanh()
        elif activation == 'linear':
            self.scale_activation = None
            LOG.debug(f'Linear Activation -> No activation function')
        else:
            raise ValueError(f'Invalid input on activation {activation}')


    def forward(self, x, x_aux):

        attention = self.aux(x_aux)
        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale == None:
            v_scale = attention
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift == None:
            v_scale = self.scale
            v_shift = attention
        else:
            raise Exception(f'Ooops, something went wrong: {self.scale}, {self.shift}')

        attention = v_scale + v_shift
        attention = self.fc(attention)

        return attention

class ModelFactory:

    def get_available_models(self):

        return ['ResNet', 'Colliot', 'SENet', 'ConcatHNN', 'ConcatHNN_V2', 'DynamicHNN', 'AttentionHNN', 'FilmHNN', 'CheckAux', 'ZeCatNet', 'ZeFIBNet', 'ZeNullNet', 'ZeNuNet']

    def get_heterogeneous_models(self):

        return self.get_available_models()[3:]

    def create_model(self, model_type='ResNet', model_args={'in_channels': 1, 'n_outputs': 1}, pretrained=None, block_grads=False):

        mdls = self.get_available_models()

        if model_type == mdls[0]:  # ResNet
            model = ResNet(**model_args)

        elif model_type == mdls[1]:  # Colliot Net
            model = Conv4_FC3()

        elif model_type == mdls[2]:  # SENet
            model = SENet(**model_args)

        elif model_type == mdls[3]:  # ConcatHNN
            model = ConcatHNN(**model_args)

        elif model_type == mdls[4]:  # ConcatHNN_V2
            model = ConcatHNN_V2(**model_args)

        elif model_type == mdls[5]:  # DynamicHNN
            model = DynamicHNN(**model_args)

        elif model_type == mdls[6]:  # AttentionHNN
            model = InteractiveHNN(**model_args)

        elif model_type == mdls[7]:  # FilmHNN
            model = FilmHNN(**model_args)

        elif model_type == mdls[8]:  # CheckAux
            model = CheckAux(**model_args)

        elif model_type == mdls[9]:  # ZeCatNet
            model = ZeCatNet(**model_args)

        elif model_type == mdls[10]:  # ZeFIBNet
            model = ZeFIBNet(**model_args)

        elif model_type == mdls[11]:  # ZeNullNet
            model = ZeNullNet(**model_args)

        elif model_type == mdls[12]:  # ZeNuNet
            model = ZeNuNet(**model_args)

        else:
            raise ValueError(model_type)

        if pretrained != None:

            pretrained_model = torch.load(pretrained)

            model_dict = model.state_dict()

            pretrained_dict = {k: v for k, v in pretrained_model.items() if 'fc' not in k and k in model_dict}
                    
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            if block_grads:
                for name, param in model.named_parameters():
                    if name in pretrained_dict:
                        param.requires_grad = False

        return model

