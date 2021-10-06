"""Module for complex functionality."""


from typing import Tuple

import torch


__all__ = [
    'cmplx_to_real',
    'real_to_cmplx',
]


def cmplx_to_real(input:torch.Tensor, dim:int) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Split a complex tensor into two real tensors which are real and imaginary parts.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor.
    dim : int
        Dimension in which the real and imaginary parts are concatenated.
        ``input.size(dim)`` must be even.

    Returns
    -------
    input_r : torch.Tensor
        Real part of ``input``.
    input_i : torch.Tensor
        Imaginary part of ``input``.
    """
    assert input.size(dim) % 2 == 0, f'input.size({dim})={input.size(dim)} must be even.'
    return torch.chunk(input, 2, dim=dim)


def real_to_cmplx(input_r:torch.Tensor, input_i:torch.Tensor, dim:int) -> torch.Tensor:
    r"""Concatenate two real tensors which are real and imaginary part into a complex tensor.

    Parameters
    ----------
    input_r : torch.Tensor
        Real part.
    input_i : torch.Tensor
        Imaginary part.
        ``input_i.size()`` must be the same as ``input_r.size()``.
    dim : int
        Dimension in which the real and imaginary parts should be concatenated.

    Returns
    -------
    input : torch.Tensor
        Complex tensor with combined ``input_r`` and ``input_i``.
    """
    assert input_r.shape == input_i.shape, f'input_r.shape={tuple(input_r.shape)} must be ' \
                                           f'the same as input_i.shape={tuple(input_i.shape)}'
    return torch.cat((input_r, input_i), dim=dim)
