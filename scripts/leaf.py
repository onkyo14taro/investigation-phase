# This script is PyTorch implementation of LEAF [1].
# The original version is implemented with TensorFlow [2].
# This script differs from the original one [2] in its API, minor computations and extensions.
# Contents not related to LEAF, implementations not mentioned in the paper [1]
# (e.g., pre-emphasis filter), etc. are excluded.
# This script adds the following extensions:
# 
# 1. Option to calculate phase features.
# 2. Another normalization option ``'l1_exact'``.
# 3. Mechanism to make the Gabor filter multi-resolution (experimental).
#
# [1] N. Zeghidour, O. Teboul, F. de Chaumont Quitry, and M. Tagliasacchi,
#     “LEAF: A Learnable Frontend for Audio Classification,”
#     in ICLR, 2021. Available: https://openreview.net/forum?id=jM76BCb6F9m.
# [2] https://github.com/google-research/leaf-audio
#     Copyright 2021 Google LLC.
#     Licensed under the Apache License, Version 2.0 (the "License").
#     http://www.apache.org/licenses/LICENSE-2.0
"""Module for LEAF."""


import collections
import math
from typing import Dict, Optional, Sequence, Tuple, Union

import einops
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from complexfunctional import real_to_cmplx, cmplx_to_real
from complexnn import ComplexBatchNorm
from featureinfo import FeatureInfoRetrieval
from phase import (
    clamp_constraint, atan2_modified, principal_angle,
    unwrap, unwrap_center_diff, interp_nan_1d,
    PulseDownsampler,
)
from specaug import spec_augment
from utils import same_padding_1d


__all__ = [
    'GaborConv1d',
    'GaussianLowpass',
    'PCENLayer',
    'SpecAugment',
    'LEAF',
]


################################################################################
################################################################################
### Helper functions
################################################################################
################################################################################
def depthwise_conv_1d(input:torch.Tensor, filter:torch.Tensor, **kwargs) -> torch.Tensor:
    r"""Depthwise 1D convolution.

    Depthwise convolution bug randomly generates NaN value in some PyTorch versions.
    This bug was probably fixed at https://github.com/pytorch/pytorch/pull/55794.
    In the near future, fixed-version will be released.

    Parameters
    ----------
    input : torch.Tensor [shape=(batch_size, in_channels, *)]
        The input tensor.
    filter : torch.Tensor [shape=(in_channels, 1)]
        The filter tensor.
    **kwargs
        Other parameters for ``conv1d`` except for ``groups``.
    """
    if not (filter.size(0) == input.size(1) and filter.size(1) == 1):
        raise ValueError(
            f'Must satisfy: filter.size(0) == input.size(1) and filter.size(1) == 1; '
            f'found input.shape={input.shape}, filter.shape={input.shape}'
        )
    return F.conv1d(input, filter, groups=input.size(1), **kwargs)


# When dynamically generating a path for feature computation,
# this class allows the following process to be repeated recursively.
### If feature B, which is necessary to calculate feature A,
### has not yet been calculated, then calculate feature B first.
class keydefaultdict(collections.defaultdict):
    r"""defaultdict with factory functions that behave differently depending on key.

    Parameters
    ----------
    default_factory : Callable[Any, ...]
        The default factory is called with a single argument to produce
        a new value when a key is not present, in ``__getitem__`` only.
        A defaultdict compares equal to a dict with the same items.
        All remaining arguments are treated the same as if they were
        passed to the dict constructor, including keyword arguments.
    """
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


################################################################################
################################################################################
### original: /leaf_audio/impulse_responses.py
### https://github.com/google-research/leaf-audio/blob/master/leaf_audio/impulse_responses.py
################################################################################
################################################################################
def _gabor_impulse_response(t:torch.Tensor, mu:torch.Tensor, sigma:torch.Tensor,
                            norm:str='l1_rough') -> torch.Tensor:
    r"""Computes the gabor impulse response.
    
    Parameters
    ----------
    t : torch.Tensor [shape=(filter_size, )]
        Time tensor (sample).
    mu : torch.Tensor [shape=(1, n_bins,)]
        Parameter for the center frequencies (rad/sample).
    sigma : torch.Tensor [shape=(n_resolutoins, n_bins)]
        Parameter for the bandwidths.
    norm : str, optional
        Norm option.
        If ``norm == 'none'``, no normalization is done.
        If ``norm == 'l1_rough'``, L1 normalization by using ``sigma``.
        since the end of the Gaussian function is truncated, it is not an exact normalization.
        If ``norm == 'l1_exact'``, L1 normalization by using the L1 norm of the output.
        By default, ``'l1_rough'``.

    Returns
    -------
    torch.Tensor [shape=(n_resolutions, n_bins, filter_size)]
        Gabor filters.
    """
    t = t[None, None, :]  # shape=(1, 1, filter_size)
    mu = mu[:, :, None]  # shape=(1, n_bins, 1)
    sigma = sigma[:, :, None]  # shape=(n_resolutions, n_bins, 1)
    gaussian = torch.exp((1.0/(2. * sigma**2)) * (-t**2))  # shape=(n_resolutions, n_bins, filter_size)
    phase = mu * t  # shape=(1, n_bins, filter_size)
    if norm == 'l1_rough':
        gaussian = gaussian / (math.sqrt(2.0 * math.pi) * sigma)
    elif norm == 'l1_exact':
        gaussian = gaussian / gaussian.sum(dim=-1, keepdims=True)
    elif norm != 'none':
        raise ValueError(f'norm="{norm}" must be either "none", "l1_rough", or "l1_exact".')
    return real_to_cmplx(
        einops.rearrange(torch.cos(phase)*gaussian, 'c f t -> (c f) t'),
        einops.rearrange(torch.sin(phase)*gaussian, 'c f t -> (c f) t'),
        dim=0
    )  # shape=(2*(n_resolutions*n_bins), filter_size); complex


def gabor_filters(mu:torch.Tensor, sigma:torch.Tensor, filter_size:int,
                  norm:str='l1_rough') -> torch.Tensor:
    r"""Computes the gabor filters from its parameters for a given size.

    Parameters
    ----------
    mu: torch.Tensor [shape=(1, n_bins,)]
        Parameter for the center frequencies (rad/sample).
    sigma: torch.Tensor [shape=(n_resolutoins, n_bins)]
        Parameter for the bandwidths.
    filter_size: int
        Size of the output filter.
    norm : str, optional
        Norm option.
        If ``norm == 'none'``, no normalization is done.
        If ``norm == 'l1_rough'``, L1 normalization by multiplying ``1 / (√(2π)σ)``.
        since the edges of the gaussian function is truncated, it is not an exact normalization.
        If ``norm == 'l1_exact'``, L1 normalization by using the L1 norm of the output.
        By default, ``'l1_rough'``.

    Returns
    -------
    torch.Tensor [shape=(n_resolutions, n_bins, filter_size)]
        Gabor filters.
    """
    # original; symmetric only when `filter_size` is odd.
    # torch.arange(-(filter_size//2), filter_size-filter_size//2).float();
    t = (torch.arange(filter_size, device=mu.device) - (filter_size-1)/2).float()  # symmetric, independent of even/odd filter_size
    return _gabor_impulse_response(t, mu=mu, sigma=sigma, norm=norm)


def gaussian_lowpass(sigma:torch.Tensor, filter_size:int, norm:str='none') -> torch.Tensor:
    r"""Generates gaussian windows centered in zero, of std sigma.

    Parameters
    ----------
    sigma: torch.Tensor [shape=(n_resolutoins, n_bins,)]
        Parameter for the bandwidth of gaussian lowpass filters.
    filter_size: int
        Size of the filters (samples).
    norm : str, optional
        Norm option.
        If ``norm == 'none'``, no normalization is done.
        If ``norm == 'l1_rough'``, L1 normalization by multiplying ``1 / (√(2π)σ)``.
        since the edges of the gaussian function is truncated, it is not an exact normalization.
        If ``norm == 'l1_exact'``, L1 normalization by using the L1 norm of the output.
        By default, ``'l1_rough'``.

    Returns
    -------
    torch.Tensor [shape=(n_resolutions, n_bins, filter_size)]
        Gaussian lowpass filters.
    """
    # Eq. (5) in the original paper, there exists a normalization term.
    # However, in the original TensorFlow implementation and the Figure A.4 in the original paper,
    # no normalization seems to be performed.
    # Therefore, no normalization was used in this PyTorch implementation either.
    sigma = sigma[:, :, None]  # shape=(n_resolutions, n_bins, 1)
    numerator = (torch.arange(filter_size, device=sigma.device) - (filter_size-1)/2).float()  # shape=(filter_size, )
    numerator = numerator[None, None, :]  # shape=(1, 1, filter_size)
    denominator = (sigma * 0.5 * (filter_size - 1))  # shape=(n_resolutions, n_bins, 1)
    filter = torch.exp(-0.5 * (numerator / denominator)**2)  # shape=(n_resolutions, n_bins, filter_size)
    if norm == 'l1_rough':
        filter = filter / (math.sqrt(2.0 * math.pi) * sigma * 0.5 * (filter_size - 1))
    elif norm == 'l1_exact':
        filter = filter / filter.sum(dim=-1, keepdims=True)
    elif filter != 'none':
        raise ValueError(f'norm="{norm}" must be either "none", "l1_rough", or "l1_exact".')
    return filter


################################################################################
################################################################################
### original: /leaf_audio/melfilters.py
### https://github.com/google-research/leaf-audio/blob/master/leaf_audio/melfilters.py
################################################################################
################################################################################
class Gabor():
    r"""This class creates gabor filters designed to match mel-filterbanks.

    Parameters
    ----------
    n_bins: int
        Number of frequency bins.
    filter_size: int
        Size of the filters (samples).
    min_freq: float
        Minimum frequency spanned by the filters.
    max_freq: float
        Maximum frequency spanned by the filters.
    sample_rate: int
        Samplerate (samples/s).
    n_fft: Optional[int]
        Number of frequency bins to compute mel-filters.
        If ``n_fft`` is None, ``n_fft`` is ``sample_rate``.
        By default, ``None``.
    """
    def __init__(self, n_bins:int=40, filter_size:int=401, n_fft:Optional[int]=None,
                 sample_rate:int=16000, min_freq:float=0., max_freq:Optional[float]=None):
        # In the original implementation, the default value of n_fft was 512.
        # The greater this value is, the smaller the computation error.
        # Therefore, in this implementation, the default value is the same as sample_rate (grater than 512),
        # since it is called only during initialization and the computation cost can be ignored.
        self.n_bins = n_bins
        self.filter_size = filter_size
        self.n_fft = n_fft if n_fft is not None else sample_rate
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.max_freq = max_freq if max_freq is not None else sample_rate/2

    def gabor_params_from_mels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Retrieves center frequencies and standard deviations of gabor filters.

        Returns
        -------
        mu : torch.Tensor [shape=(n_bins, )]
            Parameter for the center frequencies (rad/sample).
        sigma : torch.Tensor [shape=(n_bins, )]
            Parameter for the bandwidths.
        """
        coeff = math.sqrt(2. * math.log(2.)) * self.n_fft
        sqrt_filters = torch.sqrt(self.mel_filterbank())
        peak_info = torch.max(sqrt_filters, dim=1, keepdims=True)
        center_frequencies = peak_info.indices.squeeze().float()
        peaks = peak_info.values
        half_magnitudes = peaks / 2.
        fwhms = torch.sum((sqrt_filters >= half_magnitudes).float(), dim=1)
        mu = center_frequencies * (2*math.pi/self.n_fft)  # normalized angular frequency
        sigma = coeff / (math.pi * fwhms)
        return mu, sigma

    def mel_filterbank(self) -> torch.Tensor:
        r"""Creates a bank of mel-filters.

        Returns
        -------
        mel_filterbank : torch.Tensor [shape=(n_bins, 1 + n_fft//2)]
            Mel filterbank.
        """
        # build mel filter matrix
        mel_filterbank = torch.from_numpy(librosa.filters.mel(
            sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_bins,
            fmin=self.min_freq, fmax=self.max_freq, htk=True, norm=None, dtype=np.float32
        ))
        return mel_filterbank  # shape=(n_bins, 1 + n_fft//2)


################################################################################
################################################################################
### original: /leaf_audio/convolution.py, /leaf_audio/initializers.py
### https://github.com/google-research/leaf-audio/blob/master/leaf_audio/convolution.py
### https://github.com/google-research/leaf-audio/blob/master/leaf_audio/initializers.py
################################################################################
################################################################################
class GaborConv1d(nn.Module):
    r"""Implements a convolution with filters defined as complex Gabor wavelets.

    These filters are parametrized only by their center frequency and
    the full-width at half maximum of their frequency response.
    Thus, for n filters, there are 2*n parameters to learn.

    Parameters
    ----------
    n_bins : int
        Number of frequency bins.
    n_resolutions : int
        Number of resolutions.
    filter_size : int
        Size of the filter (convolution kernel).
    stride : int
        Stride of the convolution.
    padding : Union[int, str], optional
        Zero-padding added to both sides of the input.
        By default ``'same'``.
    sort_filters : bool, optional
        If set to True, sort the filters by center frequency.
        By default ``True``.
    norm : str, optional
        Norm option.
        If ``norm == 'none'``, no normalization is done.
        If ``norm == 'l1_rough'``, L1 normalization by multiplying ``1 / (√(2π)σ)``.
        since the edges of the gaussian function is truncated, it is not an exact normalization.
        If ``norm == 'l1_exact'``, L1 normalization by using the L1 norm of the output.
        By default, ``'l1_rough'``.
    min_freq : float, optional
        Minimum frequency used when initializing parameters with mel scale.
        By default 0.
    max_freq : float, optional
        Maximum frequency used when initializing parameters with mel scale.
        By default 8000.
    sample_rate : int, optional
        Sampling rate used when initializing parameters with mel scale.
        By default 16000.
    n_fft : int, optional
        The number of FFT points assumed when calculating the mel-Hz transform matrix.
        This is used when initializing parameters with mel scale.
        If set to None, ``n_fft`` will be sample_rate.
        By default ``None``.
    """
    def __init__(self, n_bins:int, n_resolutions:int,
                 filter_size:int, stride:int, padding:Union[int, str]='same',
                 sort_filters:bool=True,  # False in the original implementation
                 norm:str='l1_rough',
                 min_freq:float=0., max_freq:float=8000.,
                 sample_rate:int=16000, n_fft:Optional[int]=None):
        super().__init__()
        self.n_bins = n_bins
        self.n_resolutions = n_resolutions
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.sort_filters = sort_filters
        self.norm = norm
        self._mu = nn.Parameter(torch.Tensor(1, n_bins))
        self._sigma = nn.Parameter(torch.Tensor(n_resolutions, n_bins))
        self._initializer = Gabor(
            n_bins=n_bins, filter_size=filter_size, n_fft=n_fft,
            sample_rate=sample_rate, min_freq=min_freq, max_freq=max_freq
        )
        self.reset_parameters()

    @property
    def mu(self) -> torch.Tensor:
        r"""Constraint mu in radians.

        Mu is constrained in [0, pi] (rad/sample).
        """
        return clamp_constraint(self._mu, min=0., max=math.pi)

    @property
    def sigma(self) -> torch.Tensor:
        r"""Constraint sigma in radians.

        Sigma s.t full-width at half-maximum of the
        gaussian response is in [1, pi/2]. The full-width at half maximum of the
        Gaussian response is 2*sqrt(2*log(2))/sigma.
        See Section 2.2 of [1] for more details.

        [1] N. Zeghidour, O. Teboul, F. de Chaumont Quitry, and M. Tagliasacchi,
            “LEAF: A Learnable Frontend for Audio Classification,”
            in ICLR, 2021. Available: https://openreview.net/forum?id=jM76BCb6F9m.
        """
        sigma_lower = 4 * math.sqrt(2 * math.log(2)) / math.pi
        sigma_upper = self.filter_size * math.sqrt(2 * math.log(2)) / math.pi
        return clamp_constraint(self._sigma, min=sigma_lower, max=sigma_upper)

    def forward(self, input:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Define the computation performed at every call.

        Parameters
        ----------
        input : torch.Tensor [shape=(batch_size, 1, n_frames)]
            Audio wave.

        Returns
        -------
        output_r : torch.Tensor [shape=(batch_size, n_resolutions, n_bins, n_frames)]
            Real part of the time-frequency representation similar to a complex spectrogram.
        output_i : torch.Tensor [shape=(batch_size, n_resolutions, n_bins, n_frames)]
            Imaginary part of the time-frequency representation similar to a complex spectrogram.
        """
        filters = self.generate_filters().unsqueeze(1)  # shape=(2*(n_resolutions*n_bins), in_channels=1, width); the complex dimension is 0.
        input, padding = same_padding_1d(input, self.padding, self.filter_size, self.stride)
        output = F.conv1d(input, filters, stride=self.stride, padding=padding)  # shape=(batch, 2*(n_resolutions*n_bins), n_frames)
        output_r, output_i = cmplx_to_real(output, dim=1)  # shape=(batch, n_resolutions*n_bins, n_frames)
        output_r = einops.rearrange(output_r, 'b (c f) t -> b c f t', c=self.n_resolutions, f=self.n_bins)
        output_i = einops.rearrange(output_i, 'b (c f) t -> b c f t', c=self.n_resolutions, f=self.n_bins)
        return output_r, output_i

    def reset_parameters(self):
        mu = torch.empty_like(self.mu)
        sigma = torch.empty_like(self.sigma)
        mu[0], sigma[0] = self._initializer.gabor_params_from_mels()
        for i in range(1, self.n_resolutions):
            sigma[i] = sigma[0] * (0.5**i)
        with torch.no_grad():
            self._mu.copy_(mu)
            self._sigma.copy_(sigma)

    def generate_filters(self) -> torch.Tensor:
        r"""Generate the gabor filterbank.

        Returns
        -------
        gabor_filterbank : torch.Tensor [shape=(n_resolutions, n_bins, filter_size)]
            Gabor filterbank.
        """
        mu, sigma = self.mu, self.sigma
        if self.sort_filters:
            filter_order = torch.argsort(mu[0])
            mu = mu[:, filter_order]
            sigma = sigma[:, filter_order]
        return gabor_filters(mu, sigma, self.filter_size, norm=self.norm)  # shape=(2*(n_resolutions*n_bins), filter_size); the complex dimension is 0.

    def extra_repr(self) -> str:
        return f'n_bins={self.n_bins}, n_resolutions={self.n_resolutions}, ' \
               f'filter_size={self.filter_size}, stride={self.stride}, ' \
               f'padding={self.padding}, ' \
               f'sort_filters={self.sort_filters}'


################################################################################
################################################################################
### original: /leaf_audio/pooling.py, /leaf_audio/initializers.py
### https://github.com/google-research/leaf-audio/blob/master/leaf_audio/pooling.py
### https://github.com/google-research/leaf-audio/blob/master/leaf_audio/initializers.py
################################################################################
################################################################################
class GaussianLowpass(nn.Module):
    r"""Depthwise pooling (each input filter has its own pooling filter).

    Pooling filters are parametrized as zero-mean Gaussians, with learnable
    std. They can be initialized with 0.4 to approximate a Hanning window.

    Parameters
    ----------
    n_bins : int
        Number of frequency bins.
    n_resolutions : int
        Number of resolutions.
    filter_size : int
        Size of the filter (convolution kernel).
    stride : int
        Stride of the convolution.
        This corresponds to the so-called shift (or hop) length.
    padding : Union[int, str], optional
        Zero-padding added to both sides of the input.
        By default ``'same'``.
    trainable : bool, optional
        If set to True, the parameter sigma is trainable, by default True.
    norm : str, optional
        Norm option.
        If ``norm == 'none'``, no normalization is done.
        If ``norm == 'l1_rough'``, L1 normalization by multiplying ``1 / (√(2π)σ)``.
        since the edges of the gaussian function is truncated, it is not an exact normalization.
        If ``norm == 'l1_exact'``, L1 normalization by using the L1 norm of the output.
        By default, ``'l1_rough'``.
    init : float, optional
        Initial value of the parameter sigma, by default 0.4.
    """
    def __init__(self, n_bins:int, n_resolutions:int,
                 filter_size:int, stride:int, padding:Union[int, str]='same',
                 trainable:bool=True, norm:str='none',
                 init:float=0.4):  # <- extension
        super().__init__()
        self.n_bins = n_bins
        self.n_resolutions = n_resolutions
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding if padding == 'same' else padding
        self.init = init
        self.norm = norm
        self.trainable = trainable
        if trainable:
            self._sigma = nn.Parameter(torch.Tensor(n_resolutions, n_bins).fill_(init))
        else:
            self.register_buffer('_sigma', torch.Tensor(n_resolutions, n_bins).fill_(init))

    @property
    def sigma(self) -> torch.Tensor:
        r"""Constraint sigma in radians.

        Sigma is constrained in [``2/self.filter_size``, ``0.5``].
        """
        return clamp_constraint(self._sigma, min=2/self.filter_size, max=0.5)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        r"""Define the computation performed at every call.

        Parameters
        ----------
        input : torch.Tensor [shape=(batch_size, n_resolutions, n_bins, n_frames)]
            Time-frequency representation.

        Returns
        -------
        output : torch.Tensor [shape=(batch_size, n_resolutions, n_bins, n_frames)]
            Lowpassed time-frequency representation.
        """
        # Depthwise convolution.
        filters = self.generate_filters()
        # shape=(out_channels=n_bins*n_resolutions, in_channels/groups=1, width=filter_size)
        filters = torch.unsqueeze(einops.rearrange(filters, 'c f t -> (c f) t'), dim=1)
        input_reshaped = einops.rearrange(input, 'b c f t -> b (c f) t')
        input_reshaped, padding = same_padding_1d(
            input_reshaped, self.padding, kernel_size=self.filter_size, stride=self.stride
        )
        output = depthwise_conv_1d(input_reshaped, filters,
                                   stride=self.stride, padding=padding)
        output = einops.rearrange(output, 'b (c f) t -> b c f t',
                                  c=self.n_resolutions, f=self.n_bins)
        return output
        # # Depthwise convolution bug randomly generates NaN value.
        # # This bug was probably fixed at https://github.com/pytorch/pytorch/pull/55794.
        # # In the near future, fixed-version will be released.
        # # As a band-aid, recalculate when NaN appears.
        # for i in range(1_000):
        #     output = depthwise_conv_1d(input_reshaped, filters,
        #                                stride=self.stride, padding=padding)
        #     if not torch.isnan(output).any():
        #         output = einops.rearrange(output, 'b (c f) t -> b c f t',
        #                                   c=self.n_resolutions, f=self.n_bins)
        #         return output
        #     print(f'\rNaN bug is occuring successively {i+1:04} times.', end='')
        # raise Exception(
        #     'NaN bug occurred continuously 1,000 times. '
        #     'The program will be terminated abnormally '
        #     'since the calculation might not be completed.'
        # )

    def generate_filters(self) -> torch.Tensor:
        r"""Generate the gaussian lowpass filterbank.

        Returns
        -------
        gauss_filterbank : torch.Tensor [shape=(n_resolutions, n_bins, filter_size)]
            Gaussian lowpass filterbank.
        """
        return gaussian_lowpass(self.sigma, self.filter_size, self.norm)  # shape=(n_resolutions, n_bins, filter_size)

    def extra_repr(self) -> str:
        return f'n_bins={self.n_bins}, n_resolutions={self.n_resolutions}, ' \
               f'filter_size={self.filter_size}, stride={self.stride}, ' \
               f'padding={self.padding}, trainable={self.trainable}'


################################################################################
################################################################################
### original: /leaf_audio/postprocessing.py
### https://github.com/google-research/leaf-audio/blob/master/leaf_audio/postprocessing.py
################################################################################
################################################################################
@torch.jit.script
def _smooth(input:torch.Tensor, w:torch.Tensor, initial_state:Optional[torch.Tensor]=None) -> torch.Tensor:
    r"""Smooth a power spectrogram.

    ``y[n] = w * y[n] + (1-w) * y[n-1]``

    Parameters
    ----------
    input : torch.Tensor
        The input tensor [shape=(batch_size, n_resolutions, n_bins, n_frames)]
    w : torch.Tensor
        Coefficient of smoothing.
    initial_state : torch.Tensor [shape=(batch_size, n_resolutions, n_bins)], optional
        When inputting a sequence of temporally segmented tensors, the last frame
        of the previous tensor is given as ``initilal_state``.

    Returns
    -------
    output : torch.Tensor
        The output tensor.
    """
    if initial_state is None:
        output_frames = [input[..., 0]]
    else:
        output_frames = [w*input[..., 0] + (1-w)*initial_state]
    for i in range(1, input.size(-1)):
        output_frames.append(w*input[..., i] + (1-w)*output_frames[-1])
    return torch.stack(output_frames, dim=-1)


class ExponentialMovingAverage(nn.Module):
    r"""Computes of an exponential moving average of an sequential input.

    Parameters
    ----------
    init : float, optional
        Initial value of the smoothing coefficient.
        By default 0.4.
    per_filter : bool, optional
        Whether the smoothing should be different per channel.
        By default ``False``.
    n_bins : int, optional
        Number of frequency bins.
        If set ``per_filter`` to ``True``, this must be given.
    n_resolutions : int, optional
        Number of resolutions.
        If set ``per_filter`` to ``True``, this must be given.
    trainable : bool, optional
        If set to ``True``, the parameter sigma is trainable, by default ``True``.
    """
    def __init__(self, init:float=0.04, 
                 per_filter:bool=False,
                 n_bins:Optional[int]=None, n_resolutions:Optional[int]=None,
                 trainable:bool=False):
        super().__init__()
        self.per_filter = per_filter
        self.n_bins = n_bins
        self.n_resolutions = n_resolutions
        self.trainable = trainable
        self.init = init
        if trainable:
            if per_filter and (n_bins is None or n_resolutions is None):
                raise ValueError('If per_filter == True, n_bins and n_resolutions must be given.')
            if per_filter:
                self._weight = nn.Parameter(torch.Tensor(n_resolutions, n_bins).fill_(init))
            else:
                self._weight = nn.Parameter(torch.tensor(init).float())
        else:
            self.register_buffer('_weight', torch.tensor(init).float())

    @property
    def weight(self):
        return clamp_constraint(self._weight, min=0.0, max=1.0)

    def forward(self, input:torch.Tensor, initial_state:Optional[torch.Tensor]=None) -> torch.Tensor:
        r"""Define the computation performed at every call.

        Parameters
        ----------
        input : torch.Tensor [shape=(batch_size, n_resolutions, n_bins, n_frames)]
            Time-frequency representation.
        initial_state : torch.Tensor [shape=(batch_size, n_resolutions, n_bins)], optional
            When inputting a sequence of temporally segmented tensors, the last frame
            of the previous tensor is given as ``initilal_state``.

        Returns
        -------
        output : torch.Tensor [shape=(batch_size, n_resolutions, n_bins, n_frames)]
            Smoothed time-frequency representation.
        """
        w = self.weight  # shape=(n_resolutions, n_bins) or (, )
        w = w.unsqueeze(0) if self.per_filter else w.view(1, 1, 1)  # shape=(1, n_resolutions, n_bins) or (1, 1, 1)
        output = _smooth(input, w, initial_state)
        return output

    def extra_repr(self) -> str:
        txt = f'per_filter={self.per_filter}, '
        if self.per_filter:
            txt += f'n_bins={self.n_bins}, n_resolutions={self.n_resolutions}, '
        txt += f'trainable={self.trainable}'
        return txt


class PCENLayer(nn.Module):
    r"""Per-Channel Energy Normalization (PCEN) [1].
    This applies a fixed or learnable normalization by an exponential moving
    average smoother, and a compression.
    See [1] for more details.

    [1] Y. Wang, P. Getreuer, T. Hughes, R. F. Lyon, and R. A. Saurous,
       “Trainable frontend for robust and far-field keyword spotting,”
       in 2017 IEEE International Conference on Acoustics, Speech and Signal Processing
       (ICASSP), Mar. 2017, pp. 5670–5674.
       https://arxiv.org/abs/1607.05666

    Parameters
    ----------
    n_bins : int
        Number of frequency bins.
    n_resolutions : int
        Number of resolutions.
    alpha : float, optional
        Initial value of exponent of EMA smoother, by default 0.96.
    delta : float, optional
        Initial value of bias added before compression, by default 2.0.
    root : float, optional
        Initial value of one over exponent applied for compression
        (r in the paper), by default 2.0.
    smooth_coef : float, optional
        Initial value of the smoothing coefficient, by default 0.04.
    floor : float, optional
        Offset added to EMA smoother, by default 1e-6.
    trainable : bool, optional
        If set to True, the parameters (except for the smoothing coefficient)
        are trainable, by default False.
    smooth_coef_trainable : bool, optional
        If set to True, the smoothing coefficient is trainable, by default False.
    smooth_coef_per_filter : bool, optional
        Whether the smoothing should be different per channel.
        By default False.
    """
    def __init__(self,
                 n_bins:int,
                 n_resolutions:int,
                 alpha:float=0.96,
                 delta:float=2.0,
                 root:float=2.0,
                 smooth_coef:float=0.04,
                 floor:float=1e-6,
                 trainable:bool=False,
                 smooth_coef_trainable:bool=False,
                 smooth_coef_per_filter: bool = False):
        super().__init__()
        if floor <= 0:
            raise ValueError(f"floor={floor} must be strictly positive.")
        self.n_bins = n_bins
        self.n_resolutions = n_resolutions
        self.floor = floor
        self.trainable = trainable
        self.init_alpha = alpha
        self.init_delta = delta
        self.init_root = root
        if trainable:
            self._alpha = nn.Parameter(torch.Tensor(n_resolutions, n_bins).fill_(alpha))
            self._delta = nn.Parameter(torch.Tensor(n_resolutions, n_bins).fill_(delta))
            self._root = nn.Parameter(torch.Tensor(n_resolutions, n_bins).fill_(root))
        else:
            self.register_buffer('_alpha', torch.Tensor(n_resolutions, n_bins).fill_(alpha))
            self.register_buffer('_delta', torch.Tensor(n_resolutions, n_bins).fill_(delta))
            self.register_buffer('_root', torch.Tensor(n_resolutions, n_bins).fill_(root))
        self.ema = ExponentialMovingAverage(
            init=smooth_coef,
            per_filter=smooth_coef_per_filter,
            trainable=smooth_coef_trainable,
            n_bins=n_bins,
            n_resolutions=n_resolutions,
        )

    @property
    def alpha(self):
        return clamp_constraint(self._alpha, min=0.0, max=1.0)

    @property
    def delta(self):
        return clamp_constraint(self._delta, min=0.0)

    @property
    def root(self):
        return clamp_constraint(self._root, min=1.0)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        r"""Define the computation performed at every call.

        Parameters
        ----------
        input : torch.Tensor [shape=(batch_size, n_resolutions, n_bins, n_frames)]
            Time-frequency representation.

        Returns
        -------
        output : torch.Tensor [shape=(batch_size, n_resolutions, n_bins, n_frames)]
            Time-frequency representation after the PCEN.
        """
        # The following calculation procedure refers to the Librosa implementation.
        # https://librosa.org/doc/main/_modules/librosa/core/spectrum.html#pcen
        gain = self.alpha.unsqueeze(0).unsqueeze(-1)  # shape=(1, n_resolutions, n_bins, 1)
        bias = self.delta.unsqueeze(0).unsqueeze(-1)  # shape=(1, n_resolutions, n_bins, 1)
        power = 1. / self.root.unsqueeze(0).unsqueeze(-1)  # shape=(1, n_resolutions, n_bins, 1)
        eps = self.floor
        S = input
        S_smooth = self.ema(S)
        smooth = torch.exp(-gain * (math.log(eps) + torch.log1p(S_smooth / eps)))
        S_out = (bias ** power) * torch.expm1(power * torch.log1p(S * smooth / bias))
        return S_out

    def extra_repr(self) -> str:
        return f'n_bins={self.n_bins}, n_resolutions={self.n_resolutions}, ' \
               f'floor={self.floor}, trainable={self.trainable}'


class SpecAugment(nn.Module):
    r"""SpecAugment torch.nn.module.

    SpecAugment [1] is a data augmentation that combines three transformations:
    1. Time warping.
    2. Frequency masking.
    3. Time masking.

    [1] D. S. Park et al.,
        “SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition,”
        Proc. Interspeech 2019, pp. 2613–2617, 2019.

    Parameters
    ----------
    W : int, optional
        Time warping parameter.
        The width of the time warping is a random number in the range [0, W-1].
        Note that the maximum time warping width is ``W - 1``.
        By default 9.
    F : int, optional
        Frequency masking parameter.
        The width of the mask is a random number in the range [0, F-1].
        Note that the maximum frequency mask width is ``F - 1``.
        By default 11.
    T : int, optional
        Time masking parameter.
        The width of the mask is a random number in the range [0, T-1].
        Note that the maximum time mask width is ``T - 1``.
        By default 11.
    freq_mask_count : int, optional
        The number of times to perform frequency masking.
        By default 2.
    time_mask_count : int, optional
        The number of times to perform time masking.
        By default 2.
    mask_value : float, optional
        Value for the masked regions.
        If mask_value is None, the mean value of a given spectrogram calculated
        per batch and per channel is used as the value for the mask.
        By default, 0.0.
    """
    def __init__(self, W:int=8+1, F:int=10+1, T:int=10+1,
                 freq_mask_count:int=2, time_mask_count:int=2,
                 mask_value:float=0.0):
        super().__init__()
        self.W = W
        self.F = F
        self.T = T
        self.freq_mask_count = freq_mask_count
        self.time_mask_count = time_mask_count
        self.mask_value = mask_value

    def forward(self, spec:torch.Tensor) -> torch.Tensor:
        r"""Define the computation performed at every call.

        Parameters
        ----------
        spec : torch.Tensor [shape=(batch_size, n_resolutions, n_bins, n_frames)]
            Time-frequency representation.

        Returns
        -------
        spec : torch.Tensor [shape=(batch_size, n_resolutions, n_bins, n_frames)]
            Time-frequency representation after SpecAugment.
        """
        return spec_augment(
            spec, W=self.W, F=self.F, T=self.T,
            freq_mask_count=self.freq_mask_count,
            time_mask_count=self.time_mask_count,
            mask_value=self.mask_value
        )

    def extra_repr(self) -> str:
        return f'W={self.W}, F={self.F}, T={self.T}, ' \
               f'freq_mask_count={self.freq_mask_count}, ' \
               f'time_mask_count={self.time_mask_count}, ' \
               f'mask_value={self.mask_value}'


################################################################################
################################################################################
### original: /leaf_audio/frontend.py
### https://github.com/google-research/leaf-audio/blob/master/leaf_audio/frontend.py
################################################################################
################################################################################
class LEAF(nn.Module):
    r"""Keras layer that implements time-domain filterbanks.
    Creates a LEAF frontend, a learnable front-end that takes an audio
    waveform as input and outputs a learnable spectral representation. This layer
    can be initialized to replicate the computation of standard mel-filterbanks.
    A detailed technical description is presented in Section 3 of
    https://arxiv.org/abs/2101.08596.

    Parameters
    ----------
    n_bins : int, optional
        Number of frequency bins, by default 40.
    n_resolutions : int, optional
        Number of resolutions, by default 1.
    filter_size : int, optional
        Size of the filter (convolution kernel), by default 401.
    filter_stride : int, optional
        Stride of the lowpass filter.
        If ``downsampler == False``, ``filter_stride`` represents
        the stride of ``GaborConv1d``.
        By default 160.
    sample_rate : int, optional
        Sampling rate, by default 16000.
    features : Union[str, Sequece[str]], optional
        What features to output.
        By default, ``'power'``.
        The available features are as follows:
        * ``'power'`` : power (real)
        * ``'inst_freq'`` : instantaneous frequency (real)
        * ``'inst_freq_rot'`` : instantaneous frequency with phase rotation (real)
        * ``'grp_dly'`` : group delay of phase (real)
        * ``'grp_dly_rot'`` : group delay of phase with phase rotation (real)
        * ``'phase_phasor'`` : phasor representation of phase (complex)
        * ``'phase_phasor_rot'`` : phasor representation of phase with phase rotation (complex)
    tf_converter_sort_filters : bool, optional
        If set to ``True``, sort the filters by center frequency of ``GaborConv1d``.
        By default ``True``.
    tf_converter_norm : bool, optional
        How to normalize the norm of the filter of ``GaborConv1d``.
        If ``norm == 'none'``, no normalization is done.
        If ``norm == 'l1_rough'``, L1 normalization by using ``sigma``.
        since the end of the Gaussian function is truncated, it is not an exact normalization.
        If ``norm == 'l1_exact'``, L1 normalization by using the L1 norm of the output.
        By default, ``'l1_exact'``.
    tf_converter_min_freq : float, optional
        Minimum frequency used when initializing parameters of ``GaborConv1d``
        with mel scale.
        By default 60.0.
    tf_converter_max_freq : float, optional
        Maximum frequency used when initializing parameters of ``GaborConv1d``
        with mel scale.
        By default 7800.0.
    downsampler : bool, optional
        Whether to use the downsmaplers (``GaussianLowpass`` and ``PulseDownsampler``).
        By default ``True``.
    downsampler_init : float, optional
        Initial value of the parameter sigma of ``GaussisanLowpass``.
        By default 0.25.
    downsampler_trainable : bool, optional
        If set to True, the parameter sigma of ``GaussisanLowpass`` is trainable.
        By default ``True``.
    downsampler_norm : str, optional
        How to normalize the norm of the filter of ``GaussisanLowpass``.
        If ``norm == 'none'``, no normalization is done.
        If ``norm == 'l1_rough'``, L1 normalization by using ``sigma``.
        since the end of the Gaussian function is truncated, it is not an exact normalization.
        If ``norm == 'l1_exact'``, L1 normalization by using the L1 norm of the output.
        By default, ``'l1_exact'``.
    compressor : bool, optional
        Whether to use the compressor (``PCENLayer``).
        By default ``True``.
    compressor_alpha : float, optional
        Initial value of exponent of EMA smoother of ``PCENLayer``.
        By default 0.96.
    compressor_delta : float, optional
        Initial value of bias added before compression of ``PCENLayer``.
        By default 2.0.
    compressor_root : float, optional
        Initial value of one over exponent applied for compression
        of ``PCENLayer``.
        By default 2.0.
    compressor_smooth_coef : float, optional
        Initial value of the smoothing coefficient of ``PCENLayer``.
        By default 0.04.
    compressor_floor : float, optional
        Offset added to EMA smoother of ``PCENLayer``.
        By default 5e-8.
    compressor_trainable : float, optional
        If set to True, the parameters of ``PCENLayer``
        (except for the smoothing coefficient) are trainable.
        By default ``True``.
    compressor_smooth_coef_trainable : bool, optional
        If set to True, the smoothing coefficient of ``PCENLayer`` is trainable.
        By default ``True``.
    compressor_smooth_coef_per_filter : bool, optional
        Whether the smoothing should be different per channel for ``PCENLayer``.
        By default ``True``.
    phase_feat_attn_power : bool, optional
        If set to ``True``, multiply the phase features by
        the output power (non-negative value) after the compressor.
        By default ``False``.
    batch_norm : bool, optional
        If set to True, add two-dimensional Batch Normalization as the last output layer.
        By default ``True``.
    spec_augment : bool, optional
        If set to True, SpecAugment is applied during training.
        By default ``False``.
    spec_augment_W : int, optional
        Time warping parameter of SpecAugment.
        By default 8.
    spec_augment_F : int, optional
        Frequency masking parameter of SpecAugment.
        By default 10.
    spec_augment_T : int, optional
        Time masking parameter of SpecAugment.
        By default 10.
    spec_augment_freq_mask_count : int, optional
        The number of times to perform frequency masking for SpecAugment.
        By default 2.
    spec_augment_time_mask_count : int, optional
        The number of times to perform time masking for SpecAugment.
        By default 2.
    """
    def __init__(self, n_bins:int=40,
                 n_resolutions:int=1,
                 filter_size:int=401,
                 filter_stride:int=160,
                 sample_rate:int=16000,
                 features:Union[str, Sequence[str]]='power',
                 tf_converter_sort_filters:bool=True,  # False in the original implementation
                 tf_converter_norm:str='l1_exact',  # 'l1_rough in the original implementation
                 tf_converter_min_freq:float=60.0,
                 tf_converter_max_freq:float=7800.0,
                 downsampler:bool=True,
                 downsampler_init:float=0.4,
                 downsampler_trainable:bool=True,
                 downsampler_norm:str='l1_exact',  # None in the original implementation
                 compressor:bool=True,
                 compressor_alpha:float=0.96,
                 compressor_delta:float=2.0,
                 compressor_root:float=2.0,
                 compressor_smooth_coef:float=0.04,
                 # 1e-12 in the original implementation.
                 # In the original implementation, a clipping process was added just before
                 # the compressor to make the amplitude ≥ 1e-5 (removed in this implementation),
                 # and the gaussian lowpass filter was not normalized
                 # (if filter_size=401 and downsampler_init=0.4, the area would be about 200),
                 # this implementation uses 5e-8 = 1e-5/200 in this implementation.
                 compressor_floor:float=5e-8,
                 compressor_trainable:bool=True,
                 compressor_smooth_coef_trainable:bool=True,
                 compressor_smooth_coef_per_filter:bool=True,
                 phase_feat_attn_power:bool=False,
                 batch_norm:bool=True,
                 spec_augment:bool=False,
                 spec_augment_W:int=8+1,
                 spec_augment_F:int=10+1,
                 spec_augment_T:int=10+1,
                 spec_augment_freq_mask_count:int=2,
                 spec_augment_time_mask_count:int=2,
                 ):
        super().__init__()
        feature_info_retrieval = FeatureInfoRetrieval(features)
        self.features = feature_info_retrieval.features
        self._features_cmplx = feature_info_retrieval.find('cmplx')
        self._features_real = feature_info_retrieval.find('real')
        self.sample_rate = sample_rate
        self.out_channels \
            = (len(self._features_real) \
            + len(self._features_cmplx) * 2) * n_resolutions
        # Time-frequency representation converter
        self.tf_converter = GaborConv1d(
            n_bins=n_bins,
            n_resolutions=n_resolutions,
            filter_size=filter_size,
            stride=1 if downsampler else filter_stride,
            padding='same',
            sort_filters=tf_converter_sort_filters,
            norm=tf_converter_norm,
            min_freq=tf_converter_min_freq,
            max_freq=tf_converter_max_freq,
            sample_rate=sample_rate, n_fft=None,
        )

        if phase_feat_attn_power and ('power' not in self.features):
            features_downsampler = ['power'] + list(self.features)
        else:
            features_downsampler = self.features
        # Downsampler
        self.downsampler = nn.ModuleDict({
            feature: GaussianLowpass(
                n_bins=n_bins,
                n_resolutions=n_resolutions,
                filter_size=filter_size,
                stride=filter_stride,
                padding='same',
                trainable=downsampler_trainable,
                norm=downsampler_norm,
                init=downsampler_init)
            for feature in features_downsampler
        }) if downsampler else None
        pulse_downsampler = feature_info_retrieval.find('pulse_downsampler')
        self.pulse_downsampler = PulseDownsampler(
            filter_size=filter_size,
            stride=filter_stride,
            padding='same'
        ) if downsampler and pulse_downsampler else None

        # Power compressor
        self.compressor = PCENLayer(
            n_bins=n_bins,
            n_resolutions=n_resolutions,
            alpha=compressor_alpha,
            smooth_coef=compressor_smooth_coef,
            delta=compressor_delta,
            floor=compressor_floor,
            trainable=compressor_trainable,
            smooth_coef_trainable=compressor_smooth_coef_trainable,
            smooth_coef_per_filter=compressor_smooth_coef_per_filter,
        ) if compressor else None

        # Phase derivative compressor and attention option
        phase_deriv_features = feature_info_retrieval.find('phase_deriv')
        self.phase_feat_attn_power = phase_feat_attn_power

        # Batch normalization
        if batch_norm:
            batch_norm_modules = {}
            n_real_features = len(self._features_real)
            n_cmplx_features = len(self._features_cmplx)
            if n_real_features:
                batch_norm_modules['real'] = nn.BatchNorm2d(
                    num_features=n_resolutions*n_real_features, affine=False
                )
            if n_cmplx_features:
                batch_norm_modules['cmplx'] = ComplexBatchNorm(
                    num_features=n_resolutions*n_cmplx_features, affine=False
                )
            self.batch_norm = nn.ModuleDict(batch_norm_modules)
        else:
            self.batch_norm = None

        # SpecAugment
        self.spec_augment = SpecAugment(
            W=spec_augment_W,
            F=spec_augment_F,
            T=spec_augment_T,
            freq_mask_count=spec_augment_freq_mask_count,
            time_mask_count=spec_augment_time_mask_count
        ) if spec_augment else None

        # Buffers needed to dynamically build feature computation paths.
        # keydefaultdict makes it possible to compute dynamically the required value only once.
        self._wave = None
        self._results = keydefaultdict(self._get_feature)

    def extra_repr(self) -> str:
        return f'sample_rate={self.sample_rate}, features={self.features}, ' \
               f'phase_feat_attn_power={self.phase_feat_attn_power}'

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        r"""Define the computation performed at every call.

        1. Calculate features in ``self.features``
        2. Concatenate the calculated features into a real tensor and a complex tensor.
        3. Perform batch normalization (optional).
           * Perform real batch normalization for real features such as
             ``'power'``, ``'inst_freq'``, etc.
           * Perform complex batch normalization for complex features such as ``'phase_phasor'``.
        4. Perform SpecAugment (optional, only during training).

        Parameters
        ----------
        input : torch.Tensor [shape=(batch, 1, n_frames)]
            Audio wave.

        Returns
        -------
        tf_representation : torch.Tensor [shape=(batch, n_channels, n_bins, n_frames)]
            Time-frequency representation.
            n_channels is ``(n_real_features + 2*n_cmplx_features) * n_resolutions``.
        """
        features = self.calc_features(input)

        ### Regroup features into real tensors and complex tensors
        ############################################################
        features_post = {}
        n_cmplx_features = len(self._features_cmplx)
        n_real_features = len(self._features_real)
        if n_cmplx_features == 1:
            features_post['cmplx'] = [v for k, v in features.items()
                                      if k in self._features_cmplx][0]
        elif n_cmplx_features > 1:
            cmplx_r = []
            cmplx_i = []
            for v in (v for k, v in features.items()
                      if k in self._features_cmplx):
                v_r, v_i = cmplx_to_real(v, dim=1)
                cmplx_r.append(v_r)
                cmplx_i.append(v_i)
            features_post['cmplx'] = torch.cat(cmplx_r + cmplx_i, dim=1)
            del cmplx_r, cmplx_i, v_r, v_i
        if n_real_features == 1:
            features_post['real'] = [v for k, v in features.items()
                                     if k in self._features_real][0]
        elif n_real_features > 1:
            features_post['real'] = torch.cat([v for k, v in features.items()
                                               if k in self._features_real], dim=1)
        del features

        ### Batch normalization.
        ############################################################
        if self.batch_norm is not None:
            features_post = {k: self.batch_norm[k](v) for k, v in features_post.items()}

        ### Merges real and complex features.
        ############################################################
        if n_real_features and n_cmplx_features:  # Real and complex features
            output = torch.cat([features_post['real'], features_post['cmplx']], dim=1)
        elif n_real_features:  # Only real features
            output = features_post['real']
        else:  # Only complex features
            output = features_post['cmplx']
        del features_post

        ### SpecAugment.
        ############################################################
        if self.spec_augment is not None and self.training:
            output = self.spec_augment(output)

        return output

    @torch.cuda.amp.autocast(enabled=False)
    def calc_features(self, wave:torch.Tensor, return_all:bool=False) -> Dict[str, torch.Tensor]:
        r""" Calculate features.

        Parameters
        ----------
        input : torch.Tensor [shape=(batch, 1, n_frames)]
            Audio wave.
        return_all : bool
            If ``return_all == True``,
            return everything, including the results of the intermediate calculation process.
            If ``return_all == False``,
            return only features in ``self.features``.
            By default, ``False``.

        Returns
        -------
        features : Dict[str, torch.Tensor [shape=(batch, n_channels, n_bins, n_frames)]]
            Calculated features.
            If ``return_all == True``,
            return everything, including the results of the intermediate calculation process.
            If ``return_all == False``,
            return only features in ``self.features``.
        """
        self._wave = wave.float()
        self._results.clear()
        features = {f: self._get_feature(f) for f in self.features}
        if return_all:
            features.update({k: v for k, v in self._results.items()
                             if k not in self.features})
        self._wave = None
        self._results.clear()
        return features


    ############################################################
    ############################################################
    ### Featrue calculation.
    ############################################################
    ############################################################
    @torch.cuda.amp.autocast(enabled=False)
    def _get_feature(self, feature:str) -> torch.Tensor:
        name = f'_calc_{feature}'
        if not hasattr(self, name):
            raise ValueError(f'feature={name} is not defined.')
        return getattr(self, name)()

    @torch.cuda.amp.autocast(enabled=False)
    def _calc_cmplx_raw(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The output tensor is complex."""
        return self.tf_converter(self._wave)  # real_part, imag_part

    ############################################################
    ### Power features.
    ############################################################
    @torch.cuda.amp.autocast(enabled=False)
    def _calc_power_raw(self) -> torch.Tensor:
        """The output tensor is real."""
        cmplx_raw_r, cmplx_raw_i = self._results['cmplx_raw']
        return cmplx_raw_r**2 + cmplx_raw_i**2

    @torch.cuda.amp.autocast(enabled=False)
    def _calc_power(self) -> torch.Tensor:
        """The output tensor is real."""
        power = self._results['power_raw']
        if self.downsampler is not None:
            power = self.downsampler['power'](power)
        if self.compressor is not None:
            # The following comment-outed processing was
            # included in the original implementation.
            # In this implementation, it has been removed because we thought
            # it should be included in the floor of the PCENLayer.
            # power = torch.clamp(power, min=1e-5)
            power = self.compressor(power)
        return power

    ############################################################
    ### Phase features.
    ############################################################
    @torch.cuda.amp.autocast(enabled=False)
    def _postprocess_phase_phasor(self, feature:str) -> torch.Tensor:
        phase = self._results[f'{feature.replace("_phasor", "")}_raw']
        if self.downsampler is not None:
            # Unwrap, and then replace NaN by linear interpolation only during lowpass downsampling.
            # In order not to produce NaN in the process of back propagation,
            # return a tensor with all elements zero for convenience when all elements are NaN.
            # This process is not a problem because it will be processed by mask later.
            mask = torch.isnan(self.pulse_downsampler(phase))
            phase = self.downsampler[feature](interp_nan_1d(unwrap(phase, dim=-1)))
        else:
            mask = torch.isnan(phase)
        # If the center of the sampling point of the ``phase`` was originally NaN,
        # then the ``phase`` is regarded as undefined at the sampling points.
        # Define the value of the phasor of phase at the coordinate of
        # the phase undefined (the modulus is 0) as 0.
        phase_phasor_r = torch.cos(phase).masked_fill_(mask, 0.0)
        phase_phasor_i = torch.sin(phase).masked_fill_(mask, 0.0)
        if self.phase_feat_attn_power:
            phase_phasor_r = self._results['power']*phase_phasor_r
            phase_phasor_i = self._results['power']*phase_phasor_i
        return real_to_cmplx(phase_phasor_r, phase_phasor_i, dim=1)

    @torch.cuda.amp.autocast(enabled=False)
    def _calc_phase_raw(self) -> torch.Tensor:
        """The output tensor is real."""
        cmplx_raw_r, cmplx_raw_i = self._results['cmplx_raw']
        phase_raw = atan2_modified(cmplx_raw_i, cmplx_raw_r)
        # When the power is zero, the phase is not defined.
        # In this implementation, the undefined phase information is represented by NaN.
        return phase_raw.masked_fill_(self._results['power_raw'] == 0.0, float('nan'))

    @torch.cuda.amp.autocast(enabled=False)
    def _calc_phase_rot_raw(self) -> torch.Tensor:
        """The output tensor is real."""
        phase_raw = self._results['phase_raw']
        mu = self.tf_converter.mu.view(1, 1, -1, 1)
        stride = self.tf_converter.stride
        t = torch.arange(phase_raw.size(-1), device=mu.device) * stride
        shift = mu * t.view(1, 1, 1, -1)
        return principal_angle(phase_raw + shift)

    @torch.cuda.amp.autocast(enabled=False)
    def _calc_phase_phasor(self) -> torch.Tensor:
        """The output tensor is complex."""
        return self._postprocess_phase_phasor('phase_phasor')

    @torch.cuda.amp.autocast(enabled=False)
    def _calc_phase_phasor_rot(self) -> torch.Tensor:
        """The output tensor is complex."""
        return self._postprocess_phase_phasor('phase_phasor_rot')

    ############################################################
    ### Phase derivative features.
    ############################################################
    @torch.cuda.amp.autocast(enabled=False)
    def _common_process_phase_deriv(self, feature:str) -> torch.Tensor:
        if feature == 'inst_freq':
            phase_deriv = unwrap_center_diff(self._results['phase_raw'], dim=-1)  # (rad/sample)
        elif feature == 'inst_freq_rot':
            phase_deriv = unwrap_center_diff(self._results['phase_rot_raw'], dim=-1)  # (rad/sample)
        elif feature == 'grp_dly':
            phase_deriv = unwrap_center_diff(self._results['phase_raw'], dim=-2)  # (rad/bin)
        elif feature == 'grp_dly_rot':
            phase_deriv = unwrap_center_diff(self._results['phase_rot_raw'], dim=-2)  # (rad/bin)
        else:
            raise ValueError()
        if self.downsampler is not None:
            # Replace NaN by linear interpolation only during lowpass downsampling.
            mask = torch.isnan(self.pulse_downsampler(phase_deriv))
            phase_deriv = self.downsampler[feature](interp_nan_1d(phase_deriv))
        else:
            mask = torch.isnan(phase_deriv)
        # Define the value of the phase derivative at the coordinate
        # of the phase undefined (the modulus is 0) as 0.
        phase_deriv.masked_fill_(mask, 0.0)
        if self.phase_feat_attn_power:
            phase_deriv = self._results['power']*phase_deriv
        return phase_deriv

    @torch.cuda.amp.autocast(enabled=False)
    def _calc_inst_freq(self) -> torch.Tensor:
        """The output tensor is real."""
        return self._common_process_phase_deriv('inst_freq')

    @torch.cuda.amp.autocast(enabled=False)
    def _calc_inst_freq_rot(self) -> torch.Tensor:
        """The output tensor is real."""
        return self._common_process_phase_deriv('inst_freq_rot')

    @torch.cuda.amp.autocast(enabled=False)
    def _calc_grp_dly(self) -> torch.Tensor:
        """The output tensor is real."""
        return self._common_process_phase_deriv('grp_dly')

    @torch.cuda.amp.autocast(enabled=False)
    def _calc_grp_dly_rot(self) -> torch.Tensor:
        """The output tensor is real."""
        return self._common_process_phase_deriv('grp_dly_rot')
