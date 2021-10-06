import argparse
import csv
import os
from pathlib import Path
import random
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from const import *
from data import (
    SingleTaskDataModule
)
from systems import SingleTaskSystem


################################################################################
################################################################################
### Helper functions.
################################################################################
################################################################################
def _generate_exp_name(hparams:argparse.Namespace) -> str:
    r"""Generate the experiment name from ``hparams``.

    Parameters
    ----------
    hparams : argparse.Namespace
        Hyperparameters.
    """
    exp_name = [
        f'cat={hparams.category}',
    ]
    if hparams.category == 'single':
        exp_name.append(f'tsk={hparams.task_name}')
    exp_name += [
        f'n_bins={hparams.n_bins:03d}',
        f'feat={"-".join(hparams.features)}',
        f'attn_pow={hparams.phase_feat_attn_power}',
    ]
    return '__'.join(exp_name)


def _find_best_checkpointpath(version_dir) -> Path:
    r"""Restore from version directory.

    Parameters
    ----------
    version_dir : path-like object
        Version directory.

    Returns
    -------
    checkpoint_path : Path
        Path of the checkpoint when the accuracy was the highest.
    """
    version_dir = Path(version_dir)
    checkpoints_dir = version_dir/'checkpoints'
    metrics_file = version_dir/'metrics.csv'
    df = pd.read_csv(metrics_file)
    best_steps = df.loc[df['val__loss'] == df['val__loss'].min(), 'step'].to_list()
    checkpoint_paths = [
        p for p in sorted(checkpoints_dir.glob('step*.ckpt'))
        if int(p.stem.split('=')[1]) in best_steps
    ]
    if checkpoint_paths:
        return checkpoint_paths[-1]
    raise FileExistsError(
        f'The checkpoint file corresponding to step={best_steps} could not be found.'
    )


class RandomStateKeeper(Callback):
    r"""Keeps the random state of ``random``, ``numpy``, and ``torch``."""
    def __init__(self):
        self.loaded = False

    def on_save_checkpoint(self, trainer, pl_module, checkpoint:Dict[str, Any]) -> Dict[str, Any]:
        return {
            'random_state_random': random.getstate(),
            'random_state_numpy': np.random.get_state(),
            'random_state_torch': torch.get_rng_state(),
        }

    def on_load_checkpoint(self, callback_state:Dict[str, Any]) -> None:
        self.loaded = True
        random.setstate(callback_state['random_state_random'])
        np.random.set_state(callback_state['random_state_numpy'])
        torch.set_rng_state(callback_state['random_state_torch'])

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        if self.loaded:
            raise MisconfigurationException(
                f"When you resume from the saved random states, you must set "
                f"`num_sanity_val_steps` to 0."
            )


################################################################################
################################################################################
### Run experiment
################################################################################
################################################################################
def run(hparams:argparse.Namespace):
    r"""Run the experiment based on ``hparams``.

    Parameters
    ----------
    hparams : argparse.Namespace
        Hyperparameters.
    """
    # torch.autograd.set_detect_anomaly(True)  # Detects NaN in backpropagation.
    exp_name = _generate_exp_name(hparams)
    seed_everything(hparams.seed)  # Initialize random seeds (random, numpy, torch)
    logger = CSVLogger(DIR_RESULTS, name=exp_name, version=hparams.version)
    callbacks = [
        RandomStateKeeper(),
        ModelCheckpoint(
            os.path.join(logger.log_dir, 'checkpoints'),
            monitor='val__loss', mode='min',
            filename='{step:07d}',
            save_last=True, save_top_k=1,
        ),
    ]
    if hparams.category == 'single':
        system = SingleTaskSystem(hparams)
        datamodule = SingleTaskDataModule(
            task_name=hparams.task_name,
            batch_size=hparams.batch_size,
            num_workers=hparams.num_workers
        )
    else:
        raise ValueError(f'category={hparams.category} must be either "single"')
    if hparams.earlystopping_patience is not None:
        callbacks.append(EarlyStopping(
            monitor='val__loss', mode='min',
            patience=hparams.earlystopping_patience,
            min_delta=0.0, verbose=True,
            strict=True, check_finite=True,
            stopping_threshold=None, divergence_threshold=None,
        ))
    if hparams.version is not None:
        checkpoint_path = os.path.join(logger.log_dir, 'checkpoints', 'last.ckpt')
        metrics_prev_path = os.path.join(logger.log_dir, 'metrics.csv')
        if os.path.exists(checkpoint_path) \
            and os.path.exists(metrics_prev_path) :
            # Resume from the last epoch.
            hparams.resume_from_checkpoint = checkpoint_path
            hparams.num_sanity_val_steps = 0
            with open(metrics_prev_path, 'r') as f:
                reader = csv.DictReader(f)
                for metrics_dict in reader:
                    logger.experiment.log_metrics(metrics_dict, step=metrics_dict['step'])
    trainer = Trainer.from_argparse_args(hparams, logger=logger, callbacks=callbacks, weights_summary='full')
    os.makedirs(logger.log_dir, exist_ok=True)
    with open(os.path.join(logger.log_dir, 'model.txt'), 'w') as f:
        print(f'{ModelSummary(system, mode="full")}\n\n{system}', file=f)
    trainer.fit(system, datamodule=datamodule)
    trainer.test(ckpt_path=str(_find_best_checkpointpath(logger.log_dir)))
    if hparams.gpus:
        maximum_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        print(f'Maximum GPU memory {maximum_gpu_memory:.2f} GB')


def restore_from_version_dir(
        version_dir,
        checkpoint_state='best',
        checkpoint_path=None,
        map_location=torch.device('cpu'),
        **kwargs
    ) -> Tuple[LightningModule, LightningDataModule, Trainer]:
    r"""Restore from version directory.

    Parameters
    ----------
    version_dir : path-like object
        Version directory.
    checkpoint_state : str
        Either "best" or "last".
        If ``checkpoint_path`` is given, this argument is ignored.
        By default "best".
    checkpoint_path : path-like object
        Checkpoint file path.
        If ``checkpoint_path`` is ``None``, the checkpoint file used
        for restoring will be decided from ``checkpoint_state``.
    map_location : optional
        A function, torch.device, string or a dict specifying
        how to remap storage locations. By default torch.device('cpu').
    kwargs
        Any extra keyword args needed to init the model.
        Can also be used to override saved hyperparameter values.

    Returns
    -------
    model : LightningModule
        Model restored with best checkpoint file.
    datamodule : LightningDataModule
        LightningDataModule instance corresponding to ``model``.
    trainer : Trainer
        Trainer instance corresponding to ``model``.
    """
    version_dir = Path(version_dir)
    hparams_file = version_dir/'hparams.yaml'

    if checkpoint_path is None:
        if checkpoint_state == 'best':
            checkpoint_path = _find_best_checkpointpath(version_dir)
        elif checkpoint_state == 'last':
            checkpoint_path = version_dir/'checkpoints'/'last.ckpt'
    if not Path(checkpoint_path).exists():
        raise FileExistsError(Path(checkpoint_path))

    with open(hparams_file, 'r') as f:
        hparams = argparse.Namespace(**yaml.safe_load(f))
    if hparams.category == 'single':
        model_cls = SingleTaskSystem
        datamodule = SingleTaskDataModule(
            task_name=hparams.task_name,
            batch_size=hparams.batch_size,
            num_workers=hparams.num_workers
        )
    else:
        raise ValueError(f'category={hparams.category} must be either "single"')

    model = model_cls.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        map_location=map_location,
        hparams_file=str(hparams_file),
        **kwargs
    )
    trainer = Trainer.from_argparse_args(hparams, **kwargs)

    return model, datamodule, trainer
