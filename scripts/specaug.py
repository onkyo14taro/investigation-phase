"""Module for SpecAugment."""


from typing import Optional

import torch

from imwarp import sparse_image_warp_along_x


__all__ = [
    'spec_augment',
    'time_warp',
    'time_mask',
    'freq_mask',
]


def spec_augment(spec,
                 W:int=5, F:int=30, T:int=40,
                 freq_mask_count:int=2, time_mask_count:int=2,
                 mask_value:Optional[float]=None) -> torch.Tensor:
    r"""SpecAugment [1].

    This implementation shares the position of the mask between channels,
    but not between batches.

    [1] D. S. Park et al.,
        “SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition,”
        Proc. Interspeech 2019, pp. 2613–2617, 2019.

    Parameters
    ----------
    spec : torch.Tensor [shape=(batch_size, channels, frequency, time)]
        Spectrogram.
    W : int, optional
        Time warping parameter.
        The width of the time warping is a random number in the range [0, W-1].
        Note that the maximum time warping width is ``W - 1``.
        By default 5.
    F : int, optional
        Frequency masking parameter.
        The width of the mask is a random number in the range [0, F-1].
        Note that the maximum frequency mask width is ``F - 1``.
        By default 30.
    T : int, optional
        Time masking parameter.
        The width of the mask is a random number in the range [0, T-1].
        Note that the maximum time mask width is ``T - 1``.
        By default 40.
    freq_mask_count : int, optional
        The number of times to perform frequency masking.
        By default 2.
    time_mask_count : int, optional
        The number of times to perform time masking.
        By default 2.
    mask_value : float, optional
        Value for the masked regions.
        If ``mask_value`` is ``None``, the mean value of ``spec`` calculated
        per batch and per channel is used as the value for the mask.
        By default, ``None``.

    Returns
    -------
    torch.Tensor [shape=(batch_size, channels, frequency, time)]
        Warped and masked spectrogram.
    """
    output = time_warp(spec, W=W)
    if mask_value is None:
        # To prevent the mean value from being computed twice
        # by freq_mask() and time_mask(), precompute it.
        # Detach the mean values for the mask from the graph.
        pre_computed_mean = output.detach().mean(-1).mean(-1)  # shape=(batch, channels)
    else:
        pre_computed_mean = None
    output = freq_mask(output, F=F, count=freq_mask_count,
                       mask_value=mask_value,
                       pre_computed_mean=pre_computed_mean)
    output = time_mask(output, T=T, count=time_mask_count,
                       mask_value=mask_value,
                       pre_computed_mean=pre_computed_mean)
    return output


def time_warp(spec, W:int=5) -> torch.Tensor:
    r"""Time warping of SpecAugment [1].

    [1] D. S. Park et al.,
        “SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition,”
        Proc. Interspeech 2019, pp. 2613–2617, 2019.

    Parameters
    ----------
    spec : torch.Tensor [shape=(batch_size, channels, frequency, time)]
        Spectrogram.
    W : int, optional
        Time warping parameter.
        The width of the time warping is a random number in the range [0, W-1].
        Note that the maximum time warping width is ``W - 1``.
        By default 5.

    Returns
    -------
    time_warped_spec : torch.Tensor [shape=(batch_size, channels, frequency, time)]
    """
    if W <= 1:
        return spec
    batch_size = spec.size(0)
    n_frames = spec.size(3)

    source_control_point_locations = torch.empty(
        batch_size, 3, device=spec.device, requires_grad=False
    )
    # Time fixed locations of the source
    source_control_point_locations[:, :2] \
        = torch.tensor([0., n_frames-1.], device=spec.device, requires_grad=False).view(1, 2).float()
    # Time random locations of the source
    source_control_point_locations[:, 2] \
        = torch.randint(W, n_frames-W, size=[batch_size], device=spec.device, requires_grad=False).float()

    dest_control_point_locations = source_control_point_locations.clone()
    # Time random locations of the destination
    w_magnitude = torch.randint(0, W, size=[batch_size], device=spec.device, requires_grad=False).float()
    w_direction = torch.randint(0, 2, size=[batch_size], device=spec.device, requires_grad=False).float()
    w_direction = w_direction.masked_fill_(w_direction == 0., -1.)
    dest_control_point_locations[:, 2] += (w_direction * w_magnitude)

    warped_spec, flow_field = sparse_image_warp_along_x(
        spec, source_control_point_locations, dest_control_point_locations
    )
    return warped_spec


def freq_mask(spec:torch.Tensor, F:int=30, count:int=2,
              mask_value:Optional[float]=None,
              pre_computed_mean:Optional[torch.Tensor]=None) -> torch.Tensor:
    r"""Frequency masking of SpecAugment [1].

    This implementation shares the position of the mask between channels,
    but not between batches.

    [1] D. S. Park et al.,
        “SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition,”
        Proc. Interspeech 2019, pp. 2613–2617, 2019.

    Parameters
    ----------
    spec : torch.Tensor [shape=(batch, channels, frequency, time)]
        Spectrogram.
    F : int, optional
        Frequency masking parameter.
        The width of the mask is a random number in the range [0, F-1].
        Note that the maximum frequency mask width is ``F - 1``.
        By default 30.
    count : int, optional
        The number of times to perform frequency masking.
        By default 2.
    mask_value : float, optional
        Value for the masked regions.
        If ``mask_value`` is ``None``, the mean value of ``spec`` calculated
        per batch and per channel is used as the value for the mask.
        By default, ``None``.
    pre_computed_mean : torch.Tensor [shape=(batch_size, n_channels)], optional
        Precomputed mean value of ``spec`` per batch and per channel.
        This option is used to improve the computational efficiency in ``spec_augment()``.

    Returns
    -------
    freq_masked_spec : torch.Tensor [shape=(batch_size, n_channels, frequency, time)]
    """
    if F <= 1:
        return spec
    return _mask(spec, dim=2, param=F, count=count, mask_value=mask_value,
                 pre_computed_mean=pre_computed_mean)


def time_mask(spec:torch.Tensor, T:int=40, count:int=2,
              mask_value:Optional[float]=None,
              pre_computed_mean:Optional[torch.Tensor]=None) -> torch.Tensor:
    r"""Time masking of SpecAugment [1].

    This implementation shares the position of the mask between channels,
    but not between batches.

    [1] D. S. Park et al.,
        “SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition,”
        Proc. Interspeech 2019, pp. 2613–2617, 2019.

    Parameters
    ----------
    spec : torch.Tensor [shape=(batch_size, channels, frequency, time)]
        Spectrogram.
    T : int, optional
        Time masking parameter.
        The width of the mask is a random number in the range [0, T-1].
        Note that the maximum time mask width is ``T - 1``.
        By default 40.
    count : int, optional
        The number of times to perform time masking.
        By default 2.
    mask_value : float, optional
        Value for the masked regions.
        If ``mask_value`` is ``None``, the mean value of ``spec`` calculated
        per batch and per channel is used as the value for the mask.
        By default, ``None``.
    pre_computed_mean : torch.Tensor [shape=(batch_size, channels)], optional
        Precomputed mean value of ``spec`` per batch and per channel.
        This option is used to improve the computational efficiency in ``spec_augment()``.

    Returns
    -------
    time_masked_spec : torch.Tensor [shape=(batch_size, channels, frequency, time)]
    """
    if T <= 1:
        return spec
    return _mask(spec, dim=3, param=T, count=count, mask_value=mask_value,
                 pre_computed_mean=pre_computed_mean)


def _mask(spec:torch.Tensor, dim:int, param:int, count:int=1,
          mask_value:Optional[float]=None,
          pre_computed_mean:Optional[torch.Tensor]=None) -> torch.Tensor:
    """Masks spec.

    Parameters
    ----------
    spec : torch.Tensor [shape=(batch_size, channels, frquency, time)]
    dim : int; 2 (frequency) or 3 (time)
    param : int; Frequency or time masking parameter.
    count : int; The number of times to perform masking.
    mask_value : float, optional
    pre_computed_mean : torch.Tensor [shape=(batch_size, channels)], optional

    Returns
    -------
    masked_spec : torch.Tensor [shape=(batch_size, channels, frquency, time)]
    """
    if not (dim == 2 or dim == 3):
        raise ValueError('dim must be 2 (frequency) or 3 (time).')
    mask = _generate_mask(spec, dim=dim, param=param, count=count)
    if mask_value is not None:
        return spec.masked_fill(mask, mask_value)
    else:
        if pre_computed_mean is None:
            # If not given, computes now.
            # Detach the mean values for the mask from the graph.
            pre_computed_mean = spec.detach().mean(-1).mean(-1)  # shape=(batch, channels)
        return _mask_fill_mean(spec, mask, dim=dim,
                               pre_computed_mean=pre_computed_mean)


@torch.jit.script
def _mask_fill_mean(spec:torch.Tensor, mask:torch.Tensor, dim:int,
                    pre_computed_mean:torch.Tensor) -> torch.Tensor:
    r"""Fills elements of spec with the the mean value of spec calculated
       per batch and per channel where mask is True.

    Parameters
    ----------
    spec : torch.Tensor [shape=(batch_size, channels, frequency, time)]
    mask : torch.Tensor [shape=(batch_size, 1, frequency, 1) or (batch_size, 1, 1, time)]
    dim : int; 2 (frequency) or 3 (time)
    pre_computed_mean : torch.Tensor [shape=(batch_size, channels)]

    Returns
    -------
    masked_spec : torch.Tensor [shape=(batch_size, channels, frequency, time)]
    """
    clone = spec.clone()
    batch_size = clone.size(0)
    channels = clone.size(1)
    dim_size = clone.size(dim)
    # Batch-wise and channel-wise
    for b in range(batch_size):
        for c in range(channels):
            clone[b, c].masked_fill_(mask[b, 0], pre_computed_mean[b, c])
    return clone


def _generate_mask(spec:torch.Tensor, dim:int, param:int, count:int=1) -> torch.Tensor:
    r"""Generate a batched mask sharing the position between channels.

    Parameters
    ----------
    spec : torch.Tensor [shape=(batch_size, channels, frequency, time)]
    dim : int; 2 (frequency) or 3 (time)
    param : int; Frequency or time masking parameter.
    count : int; The number of times to perform masking.

    Returns
    -------
    mask : torch.Tensor [shape=(batch_size, 1, frequency, 1) or (batch_size, 1, 1, time)]
    """
    batch_size = spec.size(0)
    dim_size = spec.size(dim)
    output_shape = [1] * spec.ndim
    output_shape[0] = batch_size
    output_shape[dim] = dim_size
    # shape=(batch_size, frequency) or (batch_size, time)
    mask = torch.zeros(batch_size, dim_size, dtype=bool, device=spec.device, requires_grad=False)
    for _ in range(count):
        _randomize_mask_(mask, param)
    return mask.view(output_shape)  # shape=(batch_size, 1, frequency, 1) or (batch_size, 1, 1, time)


@torch.jit.script
def _randomize_mask_(mask:torch.Tensor, param:int) -> torch.Tensor:
    r"""Generate a batched mask sharing the position between channels.

    Parameters
    ----------
    mask : torch.Tensor [shape=(batch_size, frequency) or (batch_size, time)]
    param : int; Frequency or time masking parameter.

    Returns
    -------
    randomized_mask : torch.Tensor [shape=(batch_size, frequency) or (batch_size, time)]
    """
    batch_size, dim_size = mask.size()
    for b in range(batch_size):
        width = torch.randint(0, param, size=[1]).item()
        start = torch.randint(0, dim_size-width, size=[1]).item()
        mask[b, start:start+width] = True
    return mask
