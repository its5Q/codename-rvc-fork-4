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


class Flip(nn.Module):
    '''
    torch.jit.script() Compiled functions can't take variable number of arguments
    or use keyword-only arguments with defaults ~
    '''
    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x, torch.zeros([1], device=x.device)


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super(ResidualCouplingLayer, self).__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WaveNet(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=float(p_dropout),
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x, torch.zeros([1])

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()

    def __prepare_scriptable__(self):
        for hook in self.enc._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.enc)
        return self


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super(ResidualCouplingBlock, self).__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in self.flows[::-1]:
                x, _ = flow.forward(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()

    def __prepare_scriptable__(self):
        for i in range(self.n_flows):
            for hook in self.flows[i * 2]._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(self.flows[i * 2])

        return self
