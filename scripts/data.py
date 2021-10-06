import random
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pytorch_lightning import LightningDataModule

from const import *


__all__ = [
    'SingleTaskDataset',
    'SingleTaskDataModule',
]


################################################################################
################################################################################
### Helper classes and functions
################################################################################
################################################################################
def seed_worker(worker_id:int):
    r"""Function for reproducibility of multi-process loading.

    This function is given in the DataLoader argument ``worker_init_fn``.
    See [1] for details.

    [1] https://pytorch.org/docs/stable/notes/randomness.html#dataloader

    Parameters
    ----------
    worker_id : int
        Worker ID.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def eval_collate_fn(batch:List[Tuple[torch.Tensor, torch.Tensor, Dict]]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    r"""Collate-function for validation and test dataloader.

    In validation and test, the time length of the waveform differs
    from sample to sample. The function assigns information to the ``batch``
    about which index of the mini-batch a particular audio sample corresponds to.

    Parameters
    ----------
    batch : List[Tuple[torch.Tensor, torch.Tensor, Dict], ...]
        One sample of ``batch`` (e.g., ``batch[0]``) consists of the following:
        * ``wave`` : torch.Tensor [shape=(``n_stacks``, ``CROP_SAMPLES``)]
          * ``n_stacks`` depends on samples
        * ``label`` : torch.Tensor [torch.long, scalar]
        * ``properties`` : Dict[str, Any]

    Returns
    -------
    batch : Tuple[torch.Tensor, torch.Tensor, Dict]
        * ``wave`` : torch.Tensor [shape=(``sum_of_n_stacks``, ``CROP_SAMPLES``)]
          * ``sum_of_n_stacks`` is the sum of all ``n_satcks``
        * ``label`` : torch.Tensor [shape=(``batch_size``, ``CROP_SAMPLES``)]
        * ``properties`` : Dict[str, Any]
          * A key ``'stack_idx'`` has been added,
            which was not in the original ``properties``.
            ``properties['stack_idx']`` represents the offset for each sample.
            Its shape is (batch_size+1, ).
            ``properties['stack_idx'][n]`` represents start offset of n-th sample
            and stop offset of (n-1)-th sample.
    """
    wave, label, properties = list(zip(*batch))
    properties = {k: [p[k] for p in properties]
                  for k in properties[0].keys()}
    # Since each sample has a different time length
    # (shape=[n_stacks, CROP_SAMPLES], n is different),
    # create an index to identify the sample.
    stack_idx = torch.cumsum(torch.tensor([0] + [w.size(0) for w in wave]), dim=0)
    wave = torch.cat(wave, dim=0)  # shape=(sum_of_n_stacks, 1, crop_len)
    label = torch.tensor(label)  # shape=(batch_size, )
    for k, v in tuple(properties.items()):
        elm = v[0]
        if isinstance(elm, (int, float, np.number)) \
            or (isinstance(elm, np.ndarray) and elm.ndim == 0) \
            or (isinstance(elm, torch.Tensor) and elm.ndim == 0):
            properties[k] = torch.tensor(v)
        elif not isinstance(elm, str):
            # This exception is not essentially necessary. In this case,
            # this exception is set since it is expected that all values
            # except those of ``str`` are converted to torch.Tensor.
            # If you are considering non-``str`` values that are not
            # converted to torch.Tensor, you need to remove this exception.
            raise ValueError(f'Unexpected type; {type(elm)}')
    properties['stack_idx'] = stack_idx
    return wave, label, properties


################################################################################
################################################################################
### Loading audio
################################################################################
################################################################################
def random_crop(audio_filepath, crop_samples:int=CROP_SAMPLES) -> Tuple[torch.Tensor, int]:
    r"""Loads and randomly crops an audio file.

    If the length of original audio is less than ``crop_samples``,
    pads the end of the audio with zeroes to make it equal to ``crop_samples``.

    Parameters
    ----------
    audio_filepath : path-like object
        Audio file path.
    crop_samples : int, optional
        Crop length, by default ``CROP_SAMPLES``.

    Returns
    -------
    cropped_wave : torch.Tensor [shape=(crop_samples, )]
        Randomly cropped wave.
    start : int
        Start position of the crop (unit: sample).
    """
    n_samples = round(sf.info(audio_filepath).duration * SAMPLE_RATE)
    padding = max(crop_samples - n_samples, 0)
    if padding > 0:
        start = 0
        wave = sf.read(audio_filepath, dtype=np.float32)[0]
        wave = np.pad(wave, (0, padding))
    else:
        start = random.randrange(0, n_samples - crop_samples + 1)
        wave = sf.read(audio_filepath, dtype=np.float32,
                       frames=crop_samples, start=start)[0]
    return torch.from_numpy(wave), start


def stacked_crop(audio_filepath, crop_samples:int=CROP_SAMPLES) -> Tuple[torch.Tensor, int]:
    r"""Loads, crops and stacks an audio file.

    Split the audio into segments of length ``crop_samples`` length without overlap,
    and then stack them.
    If the length of a segment is less than ``crop_samples``,
    pads the end of the segment with zeroes to make it equal to ``crop_samples``.

    Parameters
    ----------
    audio_filepath : path-like object
        Audio file path.
    crop_samples : int, optional
        Crop length, by default ``CROP_SAMPLES``.

    Returns
    -------
    stacked_wave : torch.Tensor [shape=(n_stacks, crop_samples)]
        Stacked wave.
    n_stacks : int
        Number of stacks (segments).
    """
    wave = sf.read(audio_filepath, dtype=np.float32)[0]
    n_samples = wave.shape[-1]
    trailing_samples = n_samples % crop_samples
    padding = 0 if trailing_samples == 0 else crop_samples - trailing_samples
    wave = np.pad(wave, (0, padding))
    n_stacks = wave.shape[-1]//crop_samples
    wave = wave.reshape(n_stacks, crop_samples)
    return torch.from_numpy(wave), n_stacks


################################################################################
################################################################################
### Dataset
################################################################################
################################################################################
class SingleTaskDataset(Dataset):
    r"""Dataset for the single task.

    Parameters
    ----------
    task_name : str
        Task name, by default ``'SPEECH_COMMANDS'``.
    phase : str
        Training/evaludation phase.
        Either ``'train'`` or ``'eval'``.
        By default ``'train'``.
    """
    def __init__(self, task_name:str='SPEECH_COMMANDS', phase:str='train'):
        super().__init__()
        if phase not in {'train', 'eval'}:
            raise ValueError(f'phase must be either "train" or "eval"; found {phase}')
        task_info = ALL_TASK_INFO[task_name]
        self.task_name = task_name
        self.dataset_name = task_info.dataset_name
        self.target = task_info.target
        self.phase = phase
        self.list = pd.read_csv(self.dataset_path / f'{phase}.csv')

    @property
    def dataset_path(self):
        return DIR_DATASET / self.dataset_name

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        # 1. Reads a row from the table.
        properties_full = self.list.iloc[idx].to_dict()
        label = torch.tensor(properties_full[self.target])  # shape=(, );  0-dim

        # 2. Loads and crops audio.
        audio_filepath = self.dataset_path/self.phase/properties_full['audio_filename']
        if self.phase == 'train':
            wave, _ = random_crop(audio_filepath)  # shape=(self.crop_len, )
        else:
            wave, _ = stacked_crop(audio_filepath)  # shape=(n_stacks, self.crop_len)
        wave = wave.unsqueeze(-2)  # Add the dimmension of channel
        properties = {
            'task_name': self.task_name,
            'audio_filename': properties_full['audio_filename'],
            'y_true': int(properties_full[self.target]),
        }
        return wave, label, properties


################################################################################
################################################################################
### DataModule
################################################################################
################################################################################
class SingleTaskDataModule(LightningDataModule):
    r"""Datamodule for the single task.

    Parameters
    ----------
    task_name : str
        Task name, by default ``'SPEECH_COMMANDS'``.
    batch_size : int, optional
        Batch size, by default 256.
    num_workers : int, optional
        Number of workers for the data loaders, by default 1.
    """
    def __init__(self, task_name:str='SPEECH_COMMANDS',
                 batch_size:int=256, num_workers:int=1):
        super().__init__()
        self.task_name = task_name
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = SingleTaskDataset(task_name=self.task_name, phase='train')
        self.eval_set = SingleTaskDataset(task_name=self.task_name, phase='eval')

    def train_dataloader(self):
        return DataLoader(
            self.train_set, shuffle=True, worker_init_fn=seed_worker,
            batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_set, shuffle=False, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(0),
            batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=eval_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.eval_set, shuffle=False, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(0),
            batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=eval_collate_fn,
        )
