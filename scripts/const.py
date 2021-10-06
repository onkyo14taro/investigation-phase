"""Module for constant parameters."""


import dataclasses
from pathlib import Path

from utils import frozendict


__all__ = [
    'DIR_DATASET',
    'DIR_RESULTS',
    'SAMPLE_RATE',
    'FRAME_SAMPLES',
    'SHIFT_SAMPLES',
    'CROP_SAMPLES',
    'ALL_TASK_INFO',
]


################################################################################
################################################################################
### Helper classes
################################################################################
################################################################################
@dataclasses.dataclass(frozen=True)
class TaskInfo:
    dataset_name: str
    target: str
    n_classes: int


################################################################################
################################################################################
### Constants
################################################################################
################################################################################
DIR_DATASET = Path(__file__).parents[1]/'datasets'
DIR_RESULTS = Path(__file__).parents[1]/'results'

SAMPLE_RATE = 16000
FRAME_SAMPLES = 401
SHIFT_SAMPLES = 160
CROP_SAMPLES = SAMPLE_RATE  # 1 sec

ALL_TASK_INFO = frozendict(
    ACOUSTIC_SCENES=TaskInfo('TUT_urban_2018', 'label_id', 10),
    BIRD_AUDIO=TaskInfo('DCASE2018_task3_bird_audio', 'label_id', 2),
    EMOTION=TaskInfo('CREMA-D', 'label_id', 6),
    SPEAKER_VOX=TaskInfo('VoxCeleb1', 'label_id', 1_251),
    MUSIC_INST=TaskInfo('NSynth', 'inst_id', 11),
    MUSIC_PITCH=TaskInfo('NSynth', 'pitch_id', 112),
    SPEECH_COMMANDS=TaskInfo('speech_commands_v0.02', 'label_id', 35),
    LANGUAGE=TaskInfo('Voxforge', 'label_id', 6)
)
