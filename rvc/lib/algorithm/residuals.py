import torch
from itertools import chain
from typing import Optional, Tuple
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

# for resblock_s and resblock_s_mask
import torch.nn as nn
from torch.nn import Conv1d, PReLU

import torch.nn.functional as F

from rvc.lib.algorithm.wavenet import WaveNet
from rvc.lib.algorithm.commons import get_padding, init_weights

from rvc.lib.algorithm.conformer.snake_fused_triton import Snake # Fused Triton variant
from rvc.lib.algorithm.conformer.activations import SnakeBeta

LRELU_SLOPE = 0.1

class Swish(torch.nn.Module):
    def __init__(self, beta=1.0, learnable=True):
        super().__init__()
        if learnable:
            self.beta = torch.nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer("beta", torch.tensor(beta))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


def create_conv1d_layer(channels, kernel_size, dilation):
    return weight_norm(
        torch.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            1,
            dilation=dilation,
            padding=get_padding(kernel_size, dilation),
        )
    )


def apply_mask(tensor: torch.Tensor, mask: Optional[torch.Tensor]):
    return tensor * mask if mask else tensor


def apply_mask_(tensor: torch.Tensor, mask: Optional[torch.Tensor]):
    return tensor.mul_(mask) if mask else tensor


class ResBlock_PReLU(torch.nn.Module):
    """
    A residual block module that applies a series of 1D convolutional layers
    with residual connections using Parametric ReLU activation.
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int] = (1, 3, 5),
    ):
        super().__init__()
        num_layers = len(dilations)

        self.act1 = torch.nn.ModuleList([
            PReLU(num_parameters=channels, init=0.1) for _ in range(num_layers)
        ])
        self.act2 = torch.nn.ModuleList([
            PReLU(num_parameters=channels, init=0.1) for _ in range(num_layers)
        ])

        self.convs1 = self._create_convs(channels, kernel_size, dilations)
        self.convs2 = self._create_convs(channels, kernel_size, [1] * num_layers)

    @staticmethod
    def _create_convs(channels: int, kernel_size: int, dilations: Tuple[int]):
        """
        Creates a list of 1D convolutional layers with specified dilations.
        """
        layers = torch.nn.ModuleList(
            [create_conv1d_layer(channels, kernel_size, d) for d in dilations]
        )
        layers.apply(init_weights)
        return layers

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        for i, (conv1, conv2) in enumerate(zip(self.convs1, self.convs2)):

            x_residual = x

            xt = self.act1[i](x) 
            xt = apply_mask(xt, x_mask)
            xt = conv1(xt)

            xt = self.act2[i](xt)
            xt = apply_mask(xt, x_mask)
            xt = conv2(xt)

            x = xt + x_residual

            x = apply_mask(x, x_mask)

        return x

    def remove_weight_norm(self):
        for conv in chain(self.convs1, self.convs2):
            remove_weight_norm(conv)


class ResBlock_SnakeBeta(torch.nn.Module):
    """
    A residual block module that applies a series of 1D convolutional layers
    with residual connections using SnakeBeta activation.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = self._create_convs(channels, kernel_size, dilations)
        self.convs2 = self._create_convs(channels, kernel_size, [1] * len(dilations))

        # Use SnakeBeta activation functions for each layer
        self.snake_acts1 = torch.nn.ModuleList([
            SnakeBeta(channels, alpha_trainable=True, alpha_logscale=True)
            for _ in dilations
        ])
        self.snake_acts2 = torch.nn.ModuleList([
            SnakeBeta(channels, alpha_trainable=True, alpha_logscale=True)
            for _ in dilations
        ])

    @staticmethod
    def _create_convs(channels: int, kernel_size: int, dilations: Tuple[int]):
        layers = torch.nn.ModuleList(
            [create_conv1d_layer(channels, kernel_size, d) for d in dilations]
        )
        layers.apply(init_weights)
        return layers

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, self.snake_acts1, self.snake_acts2):
            x_residual = x

            xt = act1(x)  # SnakeBeta activation 1
            xt = apply_mask(xt, x_mask)
            xt = conv1(xt)

            xt = act2(xt)  # SnakeBeta activation 2
            xt = apply_mask(xt, x_mask)
            xt = conv2(xt)

            x = xt + x_residual
            x = apply_mask(x, x_mask)

        return x

    def remove_weight_norm(self):
        for conv in chain(self.convs1, self.convs2):
            remove_weight_norm(conv)


class ResBlock_Snake_Fused(torch.nn.Module):
    """
    A residual block module that applies a series of 1D convolutional layers
    with residual connections.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int] = (1, 3, 5),
    ):
        """
        Initializes the ResBlock.

        Args:
            channels (int): Number of input and output channels for the convolution layers.
            kernel_size (int): Size of the convolution kernel. Defaults to 3.
            dilations (Tuple[int]): Tuple of dilation rates for the convolution layers in the first set.
        """
        super().__init__()
        self.convs1 = self._create_convs(channels, kernel_size, dilations)
        self.convs2 = self._create_convs(channels, kernel_size, [1] * len(dilations))

        # Fused kernel Snake - Lazy load:
        from rvc.lib.algorithm.conformer.snake_fused_triton import Snake

        self.snake1 = Snake(channels, init='periodic', correction='std')
        self.snake2 = Snake(channels, init='periodic', correction='std')

    @staticmethod
    def _create_convs(channels: int, kernel_size: int, dilations: Tuple[int]):
        """
        Creates a list of 1D convolutional layers with specified dilations.

        Args:
            channels (int): Number of input and output channels for the convolution layers.
            kernel_size (int): Size of the convolution kernel.
            dilations (Tuple[int]): Tuple of dilation rates for each convolution layer.
        """
        layers = torch.nn.ModuleList(
            [create_conv1d_layer(channels, kernel_size, d) for d in dilations]
        )
        layers.apply(init_weights)
        return layers

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        for conv1, conv2 in zip(self.convs1, self.convs2):

            x_residual = x # Residual store

            xt = self.snake1(x) # Activation 1
            xt = apply_mask(xt, x_mask) # Masking 1
            xt = conv1(xt) # Conv 1

            xt = self.snake2(xt) # Activation 2
            xt = apply_mask(xt, x_mask) # Masking 2
            xt = conv2(xt) # Conv 2

            x = xt + x_residual # Residual connection

            x = apply_mask(x, x_mask) # Mask

        return x

    def remove_weight_norm(self):
        for conv in chain(self.convs1, self.convs2):
            remove_weight_norm(conv)


class ResBlock_Snake(torch.nn.Module): # Modified
    """
    A residual block module that applies a series of 1D convolutional layers
    with residual connections using Snake activation.
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int] = (1, 3, 5),
    ):

        """
        Initializes the ResBlock.

        Args:
            channels (int): Number of input and output channels for the convolution layers.
            kernel_size (int): Size of the convolution kernel. Defaults to 3.
            dilations (Tuple[int]): Tuple of dilation rates for the convolution layers in the first set.
        """
        super().__init__()
        self.convs1 = self._create_convs(channels, kernel_size, dilations)
        self.convs2 = self._create_convs(channels, kernel_size, [1] * len(dilations))
        
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs1))])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs2))])


    @staticmethod
    def _create_convs(channels: int, kernel_size: int, dilations: Tuple[int]):
        """
        Creates a list of 1D convolutional layers with specified dilations.

        Args:
            channels (int): Number of input and output channels for the convolution layers.
            kernel_size (int): Size of the convolution kernel.
            dilations (Tuple[int]): Tuple of dilation rates for each convolution layer.
        """
        layers = torch.nn.ModuleList(
            [create_conv1d_layer(channels, kernel_size, d) for d in dilations]
        )
        layers.apply(init_weights)
        return layers


    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        for conv1, conv2, a1, a2 in zip(self.convs1, self.convs2, self.alpha1, self.alpha2):

            x_residual = x # Residual store

            xt = x + (1 / a1) * (torch.sin(a1 * x) ** 2)  # Snake1D
            xt = apply_mask(xt, x_mask) # Masking 1
            xt = conv1(xt) # Conv 1

            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2) # Activation 2
            xt = apply_mask(xt, x_mask) # Masking 2
            xt = conv2(xt) # Conv 2

            x = xt + x_residual # Residual connection
            x = apply_mask(x, x_mask) # Mask

        return x

    def remove_weight_norm(self):
        for conv in chain(self.convs1, self.convs2):
            remove_weight_norm(conv)


class ResBlock(torch.nn.Module):
    """
    A residual block module that applies a series of 1D convolutional layers
    with residual connections.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int] = (1, 3, 5),
    ):
        """
        Initializes the ResBlock.

        Args:
            channels (int): Number of input and output channels for the convolution layers.
            kernel_size (int): Size of the convolution kernel. Defaults to 3.
            dilations (Tuple[int]): Tuple of dilation rates for the convolution layers in the first set.
        """
        super().__init__()
        self.convs1 = self._create_convs(channels, kernel_size, dilations)
        self.convs2 = self._create_convs(channels, kernel_size, [1] * len(dilations))

    @staticmethod
    def _create_convs(channels: int, kernel_size: int, dilations: Tuple[int]):
        """
        Creates a list of 1D convolutional layers with specified dilations.

        Args:
            channels (int): Number of input and output channels for the convolution layers.
            kernel_size (int): Size of the convolution kernel.
            dilations (Tuple[int]): Tuple of dilation rates for each convolution layer.
        """
        layers = torch.nn.ModuleList(
            [create_conv1d_layer(channels, kernel_size, d) for d in dilations]
        )
        layers.apply(init_weights)
        return layers

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        for conv1, conv2 in zip(self.convs1, self.convs2):

            x_residual = x # Residual store

            xt = torch.nn.functional.leaky_relu(x, LRELU_SLOPE) # Activation 1
            xt = apply_mask(xt, x_mask) # Masking 1
            xt = conv1(xt) # Conv 1

            xt = torch.nn.functional.leaky_relu(xt, LRELU_SLOPE) # Activation 2
            xt = apply_mask(xt, x_mask) # Masking 2
            xt = conv2(xt) # Conv 2

            x = xt + x_residual # Residual connection

            x = apply_mask(x, x_mask) # Mask

        return x

    def remove_weight_norm(self):
        for conv in chain(self.convs1, self.convs2):
            remove_weight_norm(conv)
