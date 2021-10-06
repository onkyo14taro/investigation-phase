"""Module for utilities."""

from collections import namedtuple
from typing import Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


__all__ = [
    'frozendict',
    'get_ax',
    'set_seaborn_whitegrid_ticks',
    'calculate_same_padding_1d'
    'same_padding_1d',
    'cross_entropy_smoothing',
]


def frozendict(**kwargs):
    r"""A naive implementation of frozen dict."""
    cls = namedtuple('frozendict', tuple(kwargs.keys()))
    cls.__getitem__ = lambda self, idx: getattr(self, idx)  # Overrides.
    cls.get = lambda self, idx: getattr(self, idx) if hasattr(self, idx) else None
    cls.keys = lambda self: cls._fields
    cls.values = lambda self: tuple(getattr(self, f) for f in cls._fields)
    cls.items = lambda self: tuple((f, getattr(self, f)) for f in cls._fields)
    return cls(**kwargs)


def get_ax(axes, i:int, j:int, n_rows:int, n_cols:int):
    r"""Extracts ``ax`` object according to the total number of rows and columns.

    Parameters
    ----------
    axes
        If ``n_rows == 1`` and ``n_cols == 1``, then axes is an instance of
        matplotlib.axes._subplots.AxesSubplot.
        If ``n_rows == 1`` and ``n_cols != 1``, then axes is a numpy.ndarray
        with shape=(n_cols, ).
        If ``n_rows != 1`` and ``n_cols == 1``, then axes is a numpy.ndarray
        with shape=(n_rows, ).
        If ``n_rows != 1`` and ``n_cols != 1``, then axes is a numpy.ndarray
        with shape=(n_rows, n_cols).
    i : int
        Row index.
    j : int
        Column index.
    n_rows : int
        Total number of rows.
    n_cols : int
        Total number of columns.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
    """
    if n_rows == 1 and n_cols == 1:
        return axes
    elif n_rows == 1:
        return axes[j]
    elif n_cols == 1:
        return axes[i]
    return axes[i, j]


def set_seaborn_whitegrid_ticks():
    r"""Set the style to 'seaborn-whitegrid' with outer ticks."""
    plt.style.use('seaborn-whitegrid')
    plt.rcParams["font.size"] = 12
    plt.rcParams['xtick.color'] = '.2'
    plt.rcParams['ytick.color'] = '.2'
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.size'] = 6.0
    plt.rcParams['ytick.major.size'] = 6.0
    plt.rcParams['xtick.minor.size'] = 3.0
    plt.rcParams['ytick.minor.size'] = 3.0
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True


def calculate_same_padding_1d(in_width:int, kernel_size:int, stride:int) -> Tuple[int, int]:
    r"""Calculate the size of the padding for the same case (not supported for dilation).

    Parameters
    ----------
    in_width : int
        Length of input tensor.
    kernel_size : int
        Size of the convolving kernel.
    stride : int
        Stride of the convolving kernel.

    Returns
    -------
    padding_l : int
        Left padding length.
    padding_r : int
        Right padding length.
    """
    if (in_width % stride == 0):
        padding = max(kernel_size - stride, 0)
    else:
        padding = max(kernel_size - (in_width % stride), 0)
    padding_l = padding // 2
    padding_r = padding - padding_l
    return padding_l, padding_r


def same_padding_1d(input:torch.Tensor, padding:Union[int, str],
                    kernel_size:int, stride:int) -> Tuple[torch.Tensor, int]:
    r"""Perform alternative padding handling for PyTorch.

    Parameters
    ----------
    input : torch.Tensor [shape=(batch_size, channels, length)]
        The input tensor.
    padding : Union[int, str]
        Padding parameter.
        Either length (int) or 'same'.
    kernel_size : int
        Size of the convolving kernel.
    stride : int
        Stride of the convolving kernel.

    Returns
    -------
    output : torch.Tensor
        The output tensor.
        If ``padding`` is an integer, ``output`` is ``input``.
        If ``padding`` is 'same', ``output`` is padded ``input``.
    padding_after : int
        Padding length.
        If ``padding`` is an integer, ``padding_after`` is ``padding``.
        If ``padding`` is 'same', ``padding_after`` will be ``0``.
    """
    if input.ndim != 3:
        raise ValueError(f'input.ndim={input.ndim} must be three (batch_size, in_channels, length).')
    if isinstance(padding, str) and padding != 'same':
        raise ValueError(f'padding={padding} must be either integer or "same".')
    if padding == 'same':
        # Perform padding before the convolution function.
        in_width = input.size(2)
        padding_l, padding_r = calculate_same_padding_1d(in_width, kernel_size, stride)
        # It is computationally more efficient to do padding in conv1d() than to combine pad() and conv1d().
        # Therefore, if padding_l == pdding_r, no padding is done here.
        if padding_l == padding_r:
            padding = padding_l
        else:
            input = F.pad(input, (padding_l, padding_r))
            padding = 0
        return input, padding
    else:
        return input, padding


def cross_entropy_smoothing(input:torch.Tensor, target:torch.Tensor, coef:float=0.1,
                            reduction:str='mean') -> torch.Tensor:
    r"""Calculate the cross entropy loss with label smoothing [1].

    This implementation refers to [2].

    [1] R. Müller, S. Kornblith, and G. Hinton,
        “When Does Label Smoothing Help?,” arXiv [cs.LG], Jun. 06, 2019.
    [2] https://github.com/pytorch/pytorch/issues/7455#issuecomment-869407636
    """
    input = input.log_softmax(dim=1)
    true_dist = torch.zeros_like(input)
    true_dist.scatter_(1, target.unsqueeze(1), 1.0-coef)
    true_dist += coef / input.size(1)
    if reduction == 'mean':
        return torch.mean(torch.sum(-true_dist * input, dim=1))
    elif reduction == 'sum':
        return torch.sum(torch.sum(-true_dist * input, dim=1))
    elif reduction == 'none':
        return torch.sum(-true_dist * input, dim=1)
