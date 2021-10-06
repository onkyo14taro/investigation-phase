"""Module for calculating phase features."""

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import calculate_same_padding_1d


__all__ = [
    'clamp_constraint',
    'atan2_modified',
    'principal_angle',
    'unwrap',
    'unwrap_center_diff',
    'interp_nan_1d',
    'PulseDownsampler',
]

################################################################################
################################################################################
### Functions
################################################################################
################################################################################
class ClampConstraint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input:torch.Tensor, min:float, max:float):
        return input if min is None and max is None else input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        return grad_output, None, None


@torch.jit.ignore
def clamp_constraint(input:torch.Tensor,
                      min:Optional[float]=None, max:Optional[float]=None) -> torch.Tensor:
    r"""The clamp function with the gradient defined as an identity function.

    Clamp all elements in input into the range [``min``, ``max``].
    Let min_value and max_value be ``min`` and ``max``, respectively, this returns:

    y_i = min(max(x_i, min_value), max_value)

    This function should be used only for constrain parameters.
    By nature, the derivative of the clamp function is zero in the domain being clamped.
    However, with this definition, once a parameter enters the region to be clamped,
    it will never be updated again. For this reason, the derivative in the entire domain,
    including the clamped domain, is defined as an identity function.
    This clamp function should not be used for any purpose other than to clamp parameters,
    since it does not correctly calculate back propagation.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor.
    min : float, optional
        Lower-bound of the range to be clamped to.
    max : float, optional
        Upper-bound of the range to be clamped to.

    Returns
    -------
    output : torch.Tensor
        The output tensor.
    """
    return ClampConstraint.apply(input, min, max)


class Atan2Modified(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input:torch.Tensor, other:torch.Tensor):
        # input=y, other=x
        ctx.save_for_backward(input, other)
        return torch.atan2(input, other)

    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        # It was faster without the JIT compile somehow.
        input, other = ctx.saved_tensors
        power = input**2
        power += other**2
        amin = torch.tensor(torch.finfo(power.dtype).tiny)
        torch.maximum(power, amin, out=power)  # Avoids zero-division.
        grad_input = grad_output.clone()
        grad_input /= power
        grad_other = grad_input.clone()
        grad_input *= other
        grad_other *= -input
        return grad_input, grad_other


@torch.jit.ignore
def atan2_modified(input:torch.Tensor, other:torch.Tensor) -> torch.Tensor:
    r"""The atan2 function with gradient at (0, 0) defined as a zero function.

    Element-wise arctangent of input_i / other_i with consideration of the quadrant.
    Returns a new tensor with the signed angles in radians between vector
    (other_i, input_i) and vector (1, 0). (Note that other_i, the second parameter,
    is the x-coordinate, while input_i, the first parameter, is the y-coordinate.)

    If the two arguments are both zero, the derivative of ordinary atan2 is not defined
    and back propagation is not possible.
    This function suppresses the derivative from being extremely large
    when the absolute values of both arguments are extremely small, and
    in particular, when both arguments are zero, the derivative is defined to be zero.

    Parameters
    ----------
    input : torch.Tensor
        The first input tensor.
    other : torch.Tensor
        The second input tensor.

    Returns
    -------
    output : torch.Tensor
        The output tensor.
    """
    return Atan2Modified.apply(input, other)


class PrincipalAngle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input:torch.Tensor):
        output = input.clone()
        output += math.pi
        torch.remainder(output, 2*math.pi, out=output)
        output -= math.pi
        return output.masked_fill_(output == -math.pi, math.pi)

    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        return grad_output


@torch.jit.ignore
def principal_angle(input):
    return PrincipalAngle.apply(input)


class Unwrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p:torch.Tensor, dim:int):
        torch.cuda.empty_cache()
        dd = torch.diff(p.detach(), dim=dim)
        ddmod = dd.clone()
        ddmod += math.pi
        # ddmod %= (2*math.pi)  # JIT script gives an error somehow.
        torch.remainder(ddmod, 2*math.pi, out=ddmod)
        ddmod -= math.pi
        ddmod.masked_fill_((ddmod == -math.pi) & (dd > 0), math.pi)    
        ph_correct = ddmod
        ph_correct -= dd
        ph_correct.masked_fill_(torch.abs(dd) < math.pi, 0)
        del dd, ddmod
        torch.cuda.empty_cache()
        pad = [0] * (2*abs(dim) if dim < 0 else 2*(p.ndim - dim))
        pad[-2] = 1
        ph_correct = F.pad(ph_correct, pad)
        ph_correct[torch.isnan(ph_correct)] = 0.0  # Assume that no jumps occur around NaN.
        torch.cuda.empty_cache()
        return p + ph_correct.cumsum(dim=dim)

    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        return grad_output, None, None


@torch.jit.ignore
def unwrap(p:torch.Tensor, dim:int=-1) -> torch.Tensor:
    r"""Unwrap a phase.

    To unwrap, we need a representation of the input (phase), but we consider
    the unwrapped representation to be naturally defined like a mod function,
    the error back propagation is defined as a identity function.

    The phase of the point with zero power is assumed to be represented as NaN.
    It is assumed that no unwrapping process occurs before or after NaN.

    Parameters
    ----------
    p : torch.Tensor
        Wrapped phase.
    dim : int, optional
        Dimension to compute the unwrap along.
        By default, -1.

    Returns
    -------
    unwrapped_p : torch.Tensor
        Unwrapped phase.
    """
    return Unwrap.apply(p, dim)


def _indice_along_dim(indice:torch.Tensor, ndim:int, dim:int=-1):
    indice_full = [slice(None)] * ndim
    indice_full[dim] = indice
    return indice_full


class UnwrapCenterDiff(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input:torch.Tensor, dim:int):
        output = torch.zeros_like(input)
        length = input.size(dim)
        ndim = input.ndim
        diff_mod_2pi = principal_angle(torch.diff(input, dim=dim))
        output[_indice_along_dim(torch.arange(length-1), ndim, dim)] = diff_mod_2pi
        output[_indice_along_dim(torch.arange(1,length), ndim, dim)] += diff_mod_2pi
        output[_indice_along_dim(torch.arange(1,length-1), ndim, dim)] *= 0.5
        ctx.save_for_backward(torch.tensor(ndim), torch.tensor(dim), torch.tensor(length))
        return output

    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        ndim, dim, length = ctx.saved_tensors
        grad_input = torch.empty_like(grad_output)
        grad_input[_indice_along_dim(torch.arange(2, length), ndim, dim)] \
            = grad_output[_indice_along_dim(torch.arange(1, length-1), ndim, dim)] * 0.5
        grad_input[_indice_along_dim(1, ndim, dim)] \
            = grad_output[_indice_along_dim(0, ndim, dim)]
        grad_input[_indice_along_dim(0, ndim, dim)] \
            = -grad_output[_indice_along_dim(0, ndim, dim)]
        grad_input[_indice_along_dim(torch.arange(length-2), ndim, dim)] \
            -= grad_output[_indice_along_dim(torch.arange(1, length-1), ndim, dim)] * 0.5
        grad_input[_indice_along_dim(length-2, ndim, dim)] \
            -= grad_output[_indice_along_dim(length-1, ndim, dim)]
        grad_input[_indice_along_dim(length-1, ndim, dim)] \
            += grad_output[_indice_along_dim(length-1, ndim, dim)]
        return grad_input, None


@torch.jit.ignore
def unwrap_center_diff(p:torch.Tensor, dim:int=-1):
    r"""Unwrap and calculate central difference of a phase.

    This implementation is faster than naively computing
    the central difference after unwrapping.

    The phase of the point with zero power is assumed to be represented as NaN.
    It is assumed that no unwrapping process occurs before or after NaN.

    Parameters
    ----------
    p : torch.Tensor
        Wrapped phase.
    dim : int, optional
        Dimension to compute the unwrap along.
        By default, -1.

    Returns
    -------
    diff_p : torch.Tensor
        Central difference of the unwrapped phase.
    """
    return UnwrapCenterDiff.apply(p, dim)


def interp_nan_1d(input:torch.Tensor, val_for_all_nan:float=0.0) -> torch.Tensor:
    r"""Replace NaN by using linear interpolation along the last dimension.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor.
    val_for_all_nan : float
        Value for samples where all elements are NaN.
        This value is used to avoid NaN during backpropagation.
        By default, 0.0.

    Returns
    -------
    output : torch.Tensor
        The output tensor.
    """
    mask = torch.isnan(input)
    # Shortcut (the combination of `any` and `not all` is the fastest).
    if not (torch.any(mask) and not torch.all(mask)):
        return input

    # torch.searchsorted() is slow when called repeatedly.
    # Therefore, by flattening the input to one dimension,
    # reduce the number of calls to torch.searchsorted() to only one.
    orig_size = input.size()
    length = orig_size[-1]
    input = input.flatten()
    mask = mask.flatten()
    torch.cuda.empty_cache()

    nan_indice = torch.where(mask)[0]  # shape=(X, )
    nonnan_indice = torch.where(~mask)[0]  # shape=(Y, )
    indice_indice = torch.searchsorted(nonnan_indice, nan_indice)  # shape=(X, )
    torch.cuda.empty_cache()
    nonnan_indice_l = nonnan_indice[torch.clamp(indice_indice-1, min=0)]  # shape=(X, );  Left reference indice
    torch.cuda.empty_cache()
    nonnan_indice_r = nonnan_indice[torch.clamp(indice_indice, max=nonnan_indice.size(0)-1)]  # shape=(X, );  Right reference indice

    del indice_indice
    torch.cuda.empty_cache()

    # Detect references to different samples that should not be referenced.
    orig_split = torch.div(nan_indice, length, rounding_mode='floor')
    miss_mask_l = torch.div(nonnan_indice_l, length, rounding_mode='floor') != orig_split
    miss_mask_r = torch.div(nonnan_indice_r, length, rounding_mode='floor') != orig_split

    del orig_split
    torch.cuda.empty_cache()

    # For samples where all elements are NaN, keep NaN.
    output = input.clone()
    miss_mask_both = (miss_mask_l & miss_mask_r)
    output[nan_indice[miss_mask_both]] = val_for_all_nan

    miss_mask_both_not = ~miss_mask_both

    del miss_mask_both
    torch.cuda.empty_cache()

    miss_mask_l = miss_mask_l[miss_mask_both_not]
    miss_mask_r = miss_mask_r[miss_mask_both_not]
    nan_indice = nan_indice[miss_mask_both_not]
    nonnan_indice_l = nonnan_indice_l[miss_mask_both_not]
    nonnan_indice_r = nonnan_indice_r[miss_mask_both_not]

    # Fix references to different samples that should not be referenced.
    nonnan_indice_l[miss_mask_l] = nonnan_indice_r[miss_mask_l]
    nonnan_indice_r[miss_mask_r] = nonnan_indice_l[miss_mask_r]

    del miss_mask_l, miss_mask_r, miss_mask_both_not
    torch.cuda.empty_cache()

    alpha = (   nan_indice   - nonnan_indice_l) \
          / (nonnan_indice_r - nonnan_indice_l)
    alpha.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    output[nan_indice] \
        = input[nonnan_indice_l]  * (1. - alpha) \
        + input[nonnan_indice_r] * alpha
    return output.view(*orig_size)


################################################################################
################################################################################
### Modules
################################################################################
################################################################################
class PulseDownsampler(nn.Module):
    r"""Downsample by sampling the values discrete at equal intervals.

    When downsampling, do not use a window function to smooth the data.

    Parameters
    ----------
    filter_size : int
        Size of the filter (convolution kernel).
    stride : int
        Stride of the convolution.
        This corresponds to the so-called shift (or hop) length.
    padding : Union[int, str], optional
        Zero-padding added to both sides of the input.
        By default 'same'.
    """
    def __init__(self,
                 filter_size:int,
                 stride:int,
                 padding:Union[int, str]='same'):
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding if padding == 'same' else padding

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        r"""Define the computation performed at every call.

        Parameters
        ----------
        input : torch.Tensor [shape=(batch_size, n_resolutions, n_bins, n_frames)]
            Time-frequency representation.

        Returns
        -------
        output : torch.Tensor [shape=(batch_size, n_resolutions, n_bins, n_frames_downsampled)]
            Downsampled time-frequency representation.
        """
        n_frames = input.size(-1)
        if self.padding == 'same':
            padding_l, padding_r = calculate_same_padding_1d(n_frames, self.filter_size, self.stride)
        else:
            padding_l = padding_r = self.padding
        out_width = (n_frames + (padding_l+padding_r) - (self.filter_size-1) - 1) // self.stride + 1
        indice = self.stride * torch.arange(out_width) + (self.filter_size//2) - padding_l
        n_outside_l = (indice < 0).sum()
        n_outside_r = (indice >= n_frames).sum()
        indice = indice[n_outside_l:-n_outside_r] if n_outside_r > 0 else indice[n_outside_l:]
        return F.pad(input[..., indice], (n_outside_l, n_outside_r))

    def extra_repr(self) -> str:
        return f'filter_size={self.filter_size}, stride={self.stride}, ' \
               f'padding={self.padding}'
