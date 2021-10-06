from argparse import ArgumentParser
import os
import sys

import torch
from pytorch_lightning import Trainer

from systems import _BaseTaskSystem
from train import run


os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    ######################################################################
    ### Hyperparameter setting
    ######################################################################
    parser = ArgumentParser()

    ### add PROGRAM level args
    ######################################################################

    ### add experiment level args
    ######################################################################
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--category', type=str, default='single')
    parser.add_argument('--task_name', type=str, default='SPEECH_COMMANDS')
    parser.add_argument('--earlystopping_patience', type=int)

    ### add model specific args
    ######################################################################
    parser = _BaseTaskSystem.add_model_specific_args(parser)

    ### add all the available trainer options to argparse
    ### ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    ######################################################################
    parser = Trainer.add_argparse_args(parser)

    ### parse
    ######################################################################
    hparams = parser.parse_args()
    if not torch.cuda.is_available():
        hparams.gpus = 0
        hparams.precision = 32
        hparams.amp_level = 'O0'

    ### force settings for reproducibility
    ######################################################################
    hparams.deterministic = True
    hparams.benchmark = False

    ######################################################################
    ### run the experiment
    ######################################################################
    run(hparams)


if __name__ == '__main__':
    main()
