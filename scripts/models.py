from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet import EfficientNetEncoder

from const import *
from leaf import LEAF


__all__ = [
    'SingleTaskModel'
]


####################################################################################################
####################################################################################################
### Component modules
####################################################################################################
####################################################################################################
class SingleTaskHead(nn.Module):
    r"""Single task head.

    Parameters
    ----------
    in_features : int
        Number of input parameters.
    n_classes : int
        Number of classes.
    """
    def __init__(self, in_features:int, n_classes:int):
        super().__init__()
        self.head = nn.Linear(in_features, n_classes)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        r"""Define the computation performed at every call.

        Parameters
        ----------
        input : torch.Tensor [shape=(batch, n_parameters)]
            Audio wave.

        Returns
        -------
        output : torch.Tensor [shape=(batch, n_classes)]
            Classification logit tensor.
        """
        return self.head(input)


class GlobalMaxPool2d(nn.Module):
    r"""Applies a 2D global max pooling over an input signal composed of several input planes."""
    def forward(self, input) -> torch.Tensor:
        r"""Define the computation performed at every call.

        Parameters
        ----------
        input : torch.Tensor [shape=(batch, n_channels, height, width)]

        Returns
        -------
        output : torch.Tensor [shape=(batch, n_channels, 1, 1)]
        """
        return F.adaptive_max_pool2d(input, output_size=1)


####################################################################################################
####################################################################################################
### Models
####################################################################################################
####################################################################################################
class SingleTaskModel(nn.Module):
    r"""Neural network architecture to train an audio classifier from waveforms for a single task.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    n_bins : int, optional
        Number of frequency bins, by default 40.
    features : Union[Sequence[str], str], optional
        Input features, by default ``'power'``.
    phase_feat_attn_power : bool, optional
        Whether to multiply phase features by the power after compression, by default ``False``.
    spec_augment : bool, optional
        Whether to use SpecAugment during training, by default ``False``.
    dropout_last : float, optional
        Dropout rate of the last fully connected layer for classification, by default 0.0.
    """
    def __init__(self, n_classes:int, n_bins:int=40,
                 features:Union[Sequence[str], str]='power',
                 phase_feat_attn_power:bool=False,
                 spec_augment:bool=False,
                 dropout_last:float=0.0):
        super().__init__()
        self.frontend = LEAF(
            n_bins=n_bins,
            filter_size=FRAME_SAMPLES,
            filter_stride=SHIFT_SAMPLES,
            sample_rate=SAMPLE_RATE,
            features=features,
            downsampler_init=160/(FRAME_SAMPLES-1),
            phase_feat_attn_power=phase_feat_attn_power,
            spec_augment=spec_augment,
            spec_augment_W=0,
            spec_augment_F=round(0.1 * n_bins) + 1,  # 10%
            spec_augment_T=round(0.1 * CROP_SAMPLES / SHIFT_SAMPLES) + 1,  # 10%
        )
        self.encoder = EfficientNetEncoder.from_name(
            'efficientnet-b0',
            in_channels=self.frontend.out_channels,
        )
        self.pool = nn.Sequential(
            GlobalMaxPool2d(),
            nn.Flatten(),
            nn.Dropout(dropout_last),
        )
        self.head = SingleTaskHead(self.encoder.out_channels, n_classes)

    def encode(self, input:torch.Tensor) -> torch.Tensor:
        r"""Encode the input.

        Parameters
        ----------
        input : torch.Tensor [shape=(batch, 1, n_frames)]
            Audio wave.

        Returns
        -------
        encoding : torch.Tensor [shape=(batch, encoding_dim)]
            Encoded representation.
        """
        output = self.frontend(input)
        output = self.encoder(output)
        encoding = self.pool(output)
        return encoding

    def predict(self, encoding:torch.Tensor) -> torch.Tensor:
        r"""Predict the target label (output logit).

        Parameters
        ----------
        encoding : torch.Tensor [shape=(batch, encoding_dim)]
            Encoded representation.

        Returns
        -------
        output : torch.Tensor [shape=(batch, n_classes)]
            Classification logit tensor.
        """
        return self.head(encoding)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        r"""Define the computation performed at every call.

        Parameters
        ----------
        input : torch.Tensor [shape=(batch, 1, n_frames)]
            Audio wave.

        Returns
        -------
        output : torch.Tensor [shape=(batch, n_classes)]
            Classification logit tensor.
        """
        encoding = self.encode(input)
        return self.predict(encoding)
