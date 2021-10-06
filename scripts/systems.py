from argparse import ArgumentParser
import itertools
import math
import os
from typing import Any, Dict

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl

from const import *
from leaf import gaussian_lowpass
from models import SingleTaskModel
from utils import get_ax, set_seaborn_whitegrid_ticks, cross_entropy_smoothing


__all__ = [
    'SingleTaskSystem',
]


class _BaseTaskSystem(pl.LightningModule):
    """Base class for the systems of all systems."""
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--version', type=int)
        parser.add_argument('--epoch_steps', type=int, default=10_000)
        ## Neural networks
        parser.add_argument('--n_bins', type=int, default=40)
        parser.add_argument('--features', type=str, nargs='+', default=['power'])
        parser.add_argument('--phase_feat_attn_power', action='store_true')
        ## Learning
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--lr', type=float, default=1e-4)
        ## Regularization and data augmentation
        parser.add_argument('--spec_augment', action='store_true')
        parser.add_argument('--dropout_last', type=float, default=0.0)
        parser.add_argument('--label_smooth', type=float, default=0.0)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        return parser

    def configure_optimizers(self):
        params_weight_decay = []
        params_others = []
        for module in self.model.modules():
            if isinstance(module, nn.modules.conv._ConvNd):
                params_weight_decay.append(module.weight)
                if module.bias is not None:
                    params_others.append(module.bias)
            else:
                params_others.extend([param for param in module.parameters(recurse=False)])
        optimizer = Adam([{'params': params_weight_decay, 'weight_decay': self.hparams.weight_decay},
                          {'params': params_others}],
                         lr=self.hparams.lr)
        return optimizer

    def forward(self, input):
        return self.model(input)

    def plot_learning_curve(self, metrics:Dict[str, Any]):
        """Plot the learing curve and the curves of other metrics."""
        if not os.path.exists(os.path.join(self.logger.log_dir, 'metrics.csv')):
            return

        set_seaborn_whitegrid_ticks()  # Set style.

        # Add the current validation metrics.
        metrics['step'] = self.global_step
        df = pd.read_csv(os.path.join(self.logger.log_dir, 'metrics.csv'))
        for key in metrics:
            if key not in df:
                df[key] = pd.NA
        df = df.append(metrics, ignore_index=True)
        df['step'] += 1
        # Loss; Learning curve
        loss_names_raw = ['__'.join(col.split('__')[1:])
                          for col in df.columns
                          if len(col.split('__')) >= 2 and col.split('__')[1] == 'loss']
        loss_names = []
        for loss_name in loss_names_raw:
            if loss_name not in loss_names:
                loss_names.append(loss_name)
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = plt.get_cmap('tab10')
        for i, loss_name in enumerate(loss_names):
            if f'train__{loss_name}' in df:
                train_mask = ~df[f'train__{loss_name}'].isna()
                train_plot = train_mask.any()
            else:
                train_plot = False
            if f'val__{loss_name}' in df:
                val_mask = ~df[f'val__{loss_name}'].isna()
                val_plot = val_mask.any()
            else:
                val_plot = False
            if train_plot:
                color = [c + 0.5*(1-c) for c in cm(i)]
                ax.plot(df.loc[train_mask, 'step'], df.loc[train_mask, f'train__{loss_name}'],
                        label=f'train__{loss_name}', color=color, ls='dotted')
            if val_plot:
                color = cm(i)
                ax.plot(df.loc[val_mask, 'step'], df.loc[val_mask, f'val__{loss_name}'],
                        label=f'val__{loss_name}', color=color, ls='solid')
        ax.set(title='Learning curve', xlabel='Step', ylabel='Loss')
        ax.legend(loc=(1.01, 0), frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.logger.log_dir, 'learning_curve.png'))
        plt.clf()
        plt.close()

        # Accuracy
        acc_names = (col
                     for col in df.columns
                     if col.split('__')[0] == 'acc')
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = plt.get_cmap('tab10')
        for i, acc_name in enumerate(acc_names):
            mask = ~df[acc_name].isna()
            if mask.any():
                color = cm(i)
                ax.plot(df.loc[mask, 'step'], df.loc[mask, acc_name],
                        label=acc_name, color=color, ls='solid')
        ax.set(title='Accuracy', xlabel='Step', ylabel='Accuracy')
        ax.legend(loc=(1.01, 0), frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.logger.log_dir, 'accuracy.png'))
        plt.clf()
        plt.close()

    def save_leaf_parameters(self):
        set_seaborn_whitegrid_ticks()  # Sets style.

        sample_rate = self.model.frontend.sample_rate
        n_bins = self.model.frontend.tf_converter.n_bins
        n_resolutions = self.model.frontend.tf_converter.n_resolutions
        filter_size = self.model.frontend.tf_converter.filter_size

        init_mu, init_sigma = self.model.frontend.tf_converter._initializer.gabor_params_from_mels()
        init_mu = init_mu.detach().cpu().numpy()
        init_sigma = init_sigma.detach().cpu().numpy()
        init_center_freqs = (sample_rate / (2*math.pi)) * init_mu
        init_fwhms = sample_rate * math.sqrt(2 * math.log(2)) / math.pi / init_sigma

        best_mu = self.model.frontend.tf_converter.mu.squeeze().detach().cpu().numpy()
        best_sigma = self.model.frontend.tf_converter.sigma.detach().cpu().numpy()
        best_center_freqs = (sample_rate / (2*math.pi)) * best_mu
        best_fwhms = sample_rate * math.sqrt(2 * math.log(2)) / math.pi / best_sigma

        fmin = best_center_freqs.min()
        fmax = best_center_freqs.max()
        norm = mpl.colors.Normalize(vmin=fmin, vmax=fmax)
        colors = np.array([cm.viridis(norm(f)) for f in best_center_freqs])

        results = {}

        ### Gabor filterbank
        ############################################################
        results.update({
            'GaborConv1d__mu': self.model.frontend.tf_converter.mu.detach().cpu().numpy(),
            'GaborConv1d__sigma': self.model.frontend.tf_converter.sigma.detach().cpu().numpy()
        })
        x = np.arange(1, n_bins+1)
        fig, axes = plt.subplots(1, n_resolutions, sharex=True, sharey=True,
                         figsize=(5*n_resolutions+1, 5),
                         constrained_layout=True)
        for i in range(n_resolutions):
            ax = axes if n_resolutions == 1 else axes[i]
            ax.errorbar(x, init_center_freqs, yerr=init_fwhms*(2**i)/2,
                        marker='.', linestyle='--', capsize=3, label='init')
            ax.errorbar(x, best_center_freqs, yerr=best_fwhms[i]/2,
                        marker='.', alpha=0.7, capsize=3, label='best')
            ax.set(
                title=None if n_resolutions == 1 else f'Resolution ID={i+1}',
                xlabel='Bin ID', ylabel='Frequency [Hz]',
                ylim=(
                    max(min(init_center_freqs[0] - init_fwhms[0]/2,
                            best_center_freqs[0] - best_fwhms[i, 0]/2) * 0.9,
                        1),
                    max(init_center_freqs[-1] + init_fwhms[-1]/2,
                        best_center_freqs[-1] + best_fwhms[i, -1]/2) * 1.2
                )
            )
            ax.set_yscale('log')
        ax.legend(loc=(1.01, 0), frameon=True)
        fig.suptitle('Gabor filterbank (error bars are FWHM)')
        plt.savefig(os.path.join(self.logger.log_dir, 'fig_gabor_filterbank.png'))
        plt.clf()
        plt.close()

        ### Gaussian lowpass downsampler
        ############################################################
        if self.model.frontend.downsampler is not None:
            results.update({
                f'GaussianLowpass__sigma__{k}': v.sigma.detach().cpu().numpy()
                for k, v in self.model.frontend.downsampler.items()
            })
            n_lowpass = len(self.model.frontend.downsampler)
            x = np.repeat(np.arange(filter_size).reshape(1, -1), n_bins, axis=0)
            fig, axes = plt.subplots(n_lowpass, n_resolutions, sharex=True, sharey=True,
                                     figsize=(5*n_resolutions, 2.5*n_lowpass),
                                     constrained_layout=True)
            for i, (k, v) in enumerate(self.model.frontend.downsampler.items()):
                init_lowpass = gaussian_lowpass(
                    torch.Tensor(n_resolutions, n_bins).fill_(v.init),
                    v.filter_size, norm=v.norm
                )
                for j in range(n_resolutions):
                    y = v.generate_filters()[j].detach().cpu().numpy()
                    lines = np.stack([x, y], axis=-1)
                    lc = LineCollection(lines, colors=colors)
                    ax = get_ax(axes, i, j, n_rows=n_lowpass, n_cols=n_resolutions)
                    ax.add_collection(lc)
                    ax.plot(x[0], init_lowpass[j, 0], color='black', linestyle='--', label=f'init={v.init}')
                    ax.set(
                        title=(f'Feature={k}' if n_resolutions == 1 else f'Feature={k}; Resolution ID={j+1}'),
                        xlabel='Time [sec]', ylabel='Magnitude',
                    )
                    ax.legend(frameon=True)
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, label='Frequency [Hz]')
            fig.suptitle('Gaussian Lowpass filter')
            plt.savefig(os.path.join(self.logger.log_dir, 'fig_downsampler.png'))
            plt.clf()
            plt.close()

        ### PCEN layer
        ############################################################
        results.update({
            'PCEN_alpha': self.model.frontend.compressor.alpha.detach().cpu().numpy(),
            'PCEN_delta': self.model.frontend.compressor.delta.detach().cpu().numpy(),
            'PCEN_root': self.model.frontend.compressor.root.detach().cpu().numpy(),
            'PCEN_smooth': self.model.frontend.compressor.ema.weight.detach().cpu().numpy(),
        })
        fig, axes = plt.subplots(4, n_resolutions, sharex=True,
                         figsize=(5*n_resolutions, 8),
                         constrained_layout=True)
        x = np.repeat(np.arange(1, n_bins+1)[:, None], 2, axis=1)
        for j in range(n_resolutions):
            # PCEN alpha
            ax = axes[0] if n_resolutions == 1 else axes[0, j]
            init_alpha = self.model.frontend.compressor.init_alpha
            alpha = self.model.frontend.compressor.alpha[j].detach().cpu().numpy()[:, None]
            base = np.ones_like(alpha) * init_alpha
            lines = np.stack([x, np.concatenate([base, alpha], axis=1)], axis=-1)
            lc = LineCollection(lines, colors='#aaaaaa')
            ax.plot((0, n_bins+1), (init_alpha, init_alpha), color='black', linestyle='--', linewidth=2, label='init')
            ax.add_collection(lc)
            ax.plot(lines[:, 1, 0], lines[:, 1, 1], marker='.', color='#0077ff', linestyle='None')
            ax.set(title=(r'PCEN $\alpha$' if n_resolutions == 1 else fr'PCEN $\alpha$; Resolution ID={j+1}'),
                xlabel='Bin ID', ylabel='Value')
            ax.legend(frameon=True)

            # PCEN one_over_root (r)
            ax = axes[1] if n_resolutions == 1 else axes[1, j]
            init_root = self.model.frontend.compressor.init_root
            root = self.model.frontend.compressor.root[j].detach().cpu().numpy()[:, None]
            one_over_root = 1 / root
            base = np.ones_like(root) / init_root
            lines = np.stack([x, np.concatenate([base, one_over_root], axis=1)], axis=-1)
            lc = LineCollection(lines, colors='#aaaaaa')
            ax.plot((0, n_bins+1), (1/init_root, 1/init_root), color='black', linestyle='--', linewidth=2, label='init')
            ax.add_collection(lc)
            ax.plot(lines[:, 1, 0], lines[:, 1, 1], marker='.', color='#0077ff', linestyle='None')
            ax.set(title=(r'PCEN $r$' if n_resolutions == 1 else fr'PCEN $r$; Resolution ID={j+1}'),
                xlabel='Bin ID', ylabel='Value')
            ax.legend(frameon=True)

            # PCEN delta
            ax = axes[2] if n_resolutions == 1 else axes[2, j]
            init_delta = self.model.frontend.compressor.init_delta
            delta = self.model.frontend.compressor.delta[j].detach().cpu().numpy()[:, None]
            base = np.ones_like(delta) * init_delta
            lines = np.stack([x, np.concatenate([base, delta], axis=1)], axis=-1)
            lc = LineCollection(lines, colors='#aaaaaa')
            ax.plot((0, n_bins+1), (init_delta, init_delta), color='black', linestyle='--', linewidth=2, label='init')
            ax.add_collection(lc)
            ax.plot(lines[:, 1, 0], lines[:, 1, 1], marker='.', color='#0077ff', linestyle='None')
            ax.set(title=(r'PCEN $\delta$' if n_resolutions == 1 else fr'PCEN $\delta$; Resolution ID={j+1}'),
                xlabel='Bin ID', ylabel='Value')
            ax.legend(frameon=True)

            # PCEN s
            ax = axes[3] if n_resolutions == 1 else axes[3, j]
            init_s = self.model.frontend.compressor.ema.init
            s = self.model.frontend.compressor.ema.weight[j].detach().cpu().numpy()[:, None]
            base = np.ones_like(s) * init_s
            lines = np.stack([x, np.concatenate([base, s], axis=1)], axis=-1)
            lc = LineCollection(lines, colors='#aaaaaa')
            ax.plot((0, n_bins+1), (init_s, init_s), color='black', linestyle='--', linewidth=2, label='init')
            ax.add_collection(lc)
            ax.plot(lines[:, 1, 0], lines[:, 1, 1], marker='.', color='#0077ff', linestyle='None')
            ax.set(title=(r'PCEN $s$' if n_resolutions == 1 else fr'PCEN $s$; Resolution ID={j+1}'),
                xlabel='Bin ID', ylabel='Value')
            ax.legend(frameon=True)
        fig.suptitle('PCEN parameters')
        plt.savefig(os.path.join(self.logger.log_dir, 'fig_compressor.png'))
        plt.clf()
        plt.close()

        ### Save all parameters.
        ############################################################
        np.savez(os.path.join(self.logger.log_dir, 'best_frontend.npz'), **results)


class SingleTaskSystem(_BaseTaskSystem):
    """System for the single task."""
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self._build_model()

    def _build_model(self):
        self.model = SingleTaskModel(
            ALL_TASK_INFO[self.hparams.task_name].n_classes,
            n_bins=self.hparams.n_bins,
            features=self.hparams.features,
            phase_feat_attn_power=self.hparams.phase_feat_attn_power,
            spec_augment=self.hparams.spec_augment,
            dropout_last=self.hparams.dropout_last,
        )

    def calc_training_loss(self, y_hat, y_true):
        return cross_entropy_smoothing(y_hat, y_true,
                                       coef=self.hparams.label_smooth,
                                       reduction='mean')

    def validation_test_step_process(self, batch):
        wave, y_true, properties = batch
        stack_idx = properties['stack_idx']
        assert len(stack_idx)-1 == len(y_true)
        # Splits, forwards and concatenates.
        # This is to prevent memory overflow caused by performing calculations on
        # all of the batches at once, since some samples might be very long.
        # Since the memory consumption for the same batch size is lower during evaluation
        # than during training, the maximum batch size during evaluation
        # is set to twice the batch size during training.
        n_chunks = math.ceil(wave.size(0) / (2*self.hparams.batch_size))
        y_hat = []
        for w in torch.chunk(wave, n_chunks, dim=0):
            torch.cuda.empty_cache()
            y_hat.append(self(w).detach())
        torch.cuda.empty_cache()
        y_hat = torch.cat(y_hat, dim=0)  # logit;  shape=(sum_of_n_stacks, n_classes)
        # Averages logit per sample.
        y_hat = torch.cat([
            y_hat[stack_idx[i]:stack_idx[i+1]].mean(dim=0, keepdim=True)
            for i in range(len(y_true))
        ])  # averaged logit;  shape=(n_batchs, n_classes)
        y_pred = y_hat.argmax(dim=1)  # shape=(n_batchs, )
        correct = (y_true == y_pred)  # shape=(n_batchs, )
        loss = cross_entropy_smoothing(y_hat, y_true,
                                       coef=self.hparams.label_smooth,
                                       reduction='none')  # shape=(n_batchs, );
        output = {'y_pred': y_pred.detach().cpu().numpy(),
                  'correct': correct.detach().cpu().numpy(),
                  'loss': loss.detach().cpu().numpy()}
        return output, properties

    def training_step(self, batch, batch_idx):
        wave, y_true, properties = batch
        y_hat = self(wave)
        loss = self.calc_training_loss(y_hat, y_true)
        self.log('train__loss', loss.detach().cpu().item(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output, properties = self.validation_test_step_process(batch)
        output_ = {
            'correct': output['correct'].copy(),  # np.ndarray
            'loss': output['loss'].copy(),  # np.ndarray
        }
        output.clear()
        properties.clear()
        return output_

    def validation_epoch_end(self, outputs):
        if (self.trainer is not None) and not self.trainer.sanity_checking:
            outputs = {k: [output[k] for output in outputs]
                    for k in outputs[0].keys()}
            acc = np.concatenate(outputs['correct']).mean().item()
            loss = np.concatenate(outputs['loss']).mean().item()
            metrics = {'val__loss': loss, 'acc': acc}
            self.log_dict(metrics)
            self.plot_learning_curve(metrics)

    def test_step(self, batch, batch_idx):
        output, properties = self.validation_test_step_process(batch)
        output_ = {
            'task_name': properties['task_name'].copy(),  # List[str]
            'audio_filename': properties['audio_filename'].copy(),  # List[str]
            'y_true': properties['y_true'].detach().cpu().numpy().copy(),  # np.ndarray
            'y_pred': output['y_pred'].copy(),  # np.ndarray
        }
        output.clear()
        properties.clear()
        return output_

    def test_epoch_end(self, outputs):
        self.save_leaf_parameters()
        outputs = {k: [output[k] for output in outputs]
                   for k in outputs[0].keys()}
        test_records  = pd.DataFrame({
            'task_name': list(itertools.chain(*outputs['task_name'])),
            'audio_filename': list(itertools.chain(*outputs['audio_filename'])),
            'y_true': np.concatenate(outputs['y_true']),
            'y_pred': np.concatenate(outputs['y_pred']),
        })
        test_records.to_feather(os.path.join(self.logger.log_dir, 'test_full.feather'))
        test_summary = pd.concat([
            test_records.groupby('task_name').size(),  # sample size
            test_records.groupby('task_name').apply(lambda d: (d.y_true == d.y_pred).mean()),  # accuracy
        ], axis=1)
        test_summary.columns = ['n_samples', 'acc']
        test_summary['interval95'] \
            = 1.96 \
            * ( ( (test_summary.acc*(1-test_summary.acc)) / test_summary.n_samples ) ** 0.5)
        test_summary.to_csv(os.path.join(self.logger.log_dir, 'test_summary.csv'))
