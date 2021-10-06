import collections.abc
import dataclasses
from typing import Sequence, Tuple, Union

from utils import frozendict


__all__ = [
    'AVAILABLE_FEATURES_INFO',
    'FeatureInfoRetrieval',
]


################################################################################
################################################################################
### Helper functions and classes
################################################################################
################################################################################
@dataclasses.dataclass(frozen=True)
class FeatureInfo:
    cmplx: bool
    phase_deriv: bool
    pulse_downsampler: bool
    @property
    def real(self):
        return not self.cmplx if self.cmplx is not None else None


################################################################################
################################################################################
### Main functions and classes
################################################################################
################################################################################
AVAILABLE_FEATURES_INFO = frozendict(
    power=FeatureInfo(
        cmplx=False,
        phase_deriv=False,
        pulse_downsampler=False,
    ),
    phase_phasor=FeatureInfo(
        cmplx=True,
        phase_deriv=False,
        pulse_downsampler=True,
    ),
    phase_phasor_rot=FeatureInfo(
        cmplx=True,
        phase_deriv=False,
        pulse_downsampler=True,
    ),
    inst_freq=FeatureInfo(
        cmplx=False,
        phase_deriv=True,
        pulse_downsampler=True,
    ),
    inst_freq_rot=FeatureInfo(
        cmplx=False,
        phase_deriv=True,
        pulse_downsampler=True,
    ),
    grp_dly=FeatureInfo(
        cmplx=False,
        phase_deriv=True,
        pulse_downsampler=True,
    ),
    grp_dly_rot=FeatureInfo(
        cmplx=False,
        phase_deriv=True,
        pulse_downsampler=True,
    ),
)


def _validates_and_formats_features(features:Union[str, Sequence[str]]) -> Tuple[str, ...]:
    r"""Validates the value of ``features``.

    Parameters
    ----------
    features : Union[str, Sequence[str]]
        Feature names.

    Returns
    -------
    validated_features : Tuple[str, ...]
        Validated and converted to a tuple[str] ``features``.
    """
    _err_msg = '`features` must be instance of str or Sequence[str]; found {}'

    # Validates type.
    if isinstance(features, str):
        features = (features,)
    elif isinstance(features, collections.abc.Iterable):
        features = tuple(features)
        if not features:
            raise ValueError(f'``features={features}`` must not be empty.')
        if not all(isinstance(_, str) for _ in features):
            raise ValueError(_err_msg.format(features))
    else:
        raise ValueError(_err_msg.format(features))

    available_features = set(AVAILABLE_FEATURES_INFO._fields)
    if set(features).difference(available_features):
        raise ValueError(f'`features` must be in {available_features}')
    return features


class FeatureInfoRetrieval:
    r"""Feature information retrieval.

    Parameters
    ----------
    features : Union[str, Sequence[str]]
        Names of features for which you want to retrieve information.
    """
    def __init__(self, features:Union[str, Sequence[str]]):
        self.features = _validates_and_formats_features(features)

    def find(self, attr:str) -> Tuple[str, ...]:
        r"""Find features that have specific attributes.

        Parameters
        ----------
        attr : str
            Parameters you want to find.

        Returns
        -------
        features_with_attr : Tuple[str, ...]
            Tuple of features with ``attr``.
        """
        if not hasattr(AVAILABLE_FEATURES_INFO[self.features[0]], attr):
            raise ValueError(f'attr={attr} does not exist.')
        return tuple((f for f in self.features
                      if getattr(AVAILABLE_FEATURES_INFO[f], attr)))

    def __str__(self) -> str:
        return f'FeatureInfoRetrieval(features={self.features}, ' \
               f'features_dependencies={self.features_dependencies})'

    def __repr__(self) -> str:
        return str(self)
