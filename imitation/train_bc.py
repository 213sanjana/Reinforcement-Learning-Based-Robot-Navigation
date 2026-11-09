"""Supervised behavior cloning; validation loss selects checkpoints, never test data."""
import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml
from imitation.bc_dataset import load_training_pair
from imitation.bc_network import ARCHITECTURE, BCNetwork, resolve_device

LOG = logging.getLogger(__name__)


def seed_everything(seed, deterministic=True):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class EarlyStopping:
    def __init__(self, patience, min_delta=0.):
        self.patience, self.min_delta = patience, min_delta
        self.reference = float('inf')
        self.stale = 0

    def update(self, loss):
        if loss < self.reference - self.min_delta:
            self.reference, self.stale = loss, 0
        else:
            self.stale += 1
        return self.stale >= self.patience


def run_epoch(network, loader, device, optimizer=None, gradient_clip=1.):
    training = optimizer is not None
    network.train(training)
    total_loss, total_correct, count = 0., 0, 0
    with torch.set_grad_enabled(training):
        for observations, actions in loader:
            observations = observations.to(device, non_blocking=device.type == 'cuda')
            actions = actions.to(device, non_blocking=device.type == 'cuda')
            logits = network(observations)
            loss = nn.functional.cross_entropy(logits, actions)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite cross-entropy loss')
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), gradient_clip)
                optimizer.step()
            batch = len(actions)
            total_loss += float(loss.detach().item()) * batch
            total_correct += int((logits.detach().argmax(1) == actions).sum().item())
            count += batch
    if count == 0:
        raise ValueError('Empty data loader')
    return total_loss/count, total_correct/count


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def plot_metrics(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11,4), constrained_layout=True)
    for prefix, label in (('train','Training'), ('validation','Validation')):
        axes[0].plot([r['epoch'] for r in rows], [r[prefix+'_loss'] for r in rows], label=label)
        axes[1].plot([r['epoch'] for r in rows], [r[prefix+'_accuracy'] for r in rows], label=label)
    axes[0].set(ylabel='Cross-entropy', xlabel='Epoch', title='Action classification loss')
    axes[1].set(ylabel='Action accuracy', xlabel='Epoch', ylim=(0,1), title='Expert action agreement')
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def validate_config(cfg):
    if set(cfg.get('data', {})) != {'train','validation'}:
        raise ValueError('data must contain only train and validation paths; no test input')
    t = cfg['training']
    for key in ('batch_size','epochs','patience'):
        if not isinstance(t[key], int) or t[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if t['num_workers'] < 0 or t['learning_rate'] <= 0 or t['gradient_clip'] <= 0 or t['min_delta'] < 0:
        raise ValueError('Invalid training settings')
    if t['seed'] < 0:
        raise ValueError('seed must be nonnegative')


def train(cfg):
    validate_config(cfg)
    settings = cfg['training']
    output = Path(settings['output_directory'])
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new output_directory')
    device = resolve_device(settings['device'])
    seed_everything(settings['seed'], settings['deterministic'])
    training, validation = load_training_pair(cfg['data']['train'], cfg['data']['validation'])
    network_config = dict(height=training.environment['height'], width=training.environment['width'],
                          hidden_size=cfg['model']['hidden_size'], n_actions=4)
    network = BCNetwork(**network_config).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=settings['learning_rate'])
    generator = torch.Generator().manual_seed(settings['seed'])
    val_generator = torch.Generator().manual_seed(settings['seed'] + 1)
    common = dict(batch_size=settings['batch_size'], num_workers=settings['num_workers'],
                  pin_memory=device.type == 'cuda', worker_init_fn=seed_worker)
    train_loader = DataLoader(training, shuffle=True, generator=generator, **common)
    validation_loader = DataLoader(validation, shuffle=False, generator=val_generator, **common)
    # Exclusive directory ownership: all checkpoint updates belong to this new run.
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf8')
    provenance = dict(train={'path':str(training.path), 'sha256':training.sha256, 'episodes':training.episodes,
                            'transitions':len(training)},
                      validation={'path':str(validation.path), 'sha256':validation.sha256,
                                  'episodes':validation.episodes, 'transitions':len(validation)})
    best_loss, best_epoch = float('inf'), 0
    stopper = EarlyStopping(settings['patience'], settings['min_delta'])
    rows = []
    LOG.info('BC device=%s train=%d validation=%d parameters=%d', device, len(training), len(validation),
             sum(p.numel() for p in network.parameters()))
    fields = ['epoch','train_loss','train_accuracy','validation_loss','validation_accuracy',
              'learning_rate','epoch_seconds','best_epoch','early_stop_stale']
    started = time.perf_counter()
    with (output/'metrics.csv').open('x', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); stream.flush()
        for epoch in range(1, settings['epochs']+1):
            epoch_start = time.perf_counter()
            train_loss, train_accuracy = run_epoch(network, train_loader, device, optimizer, settings['gradient_clip'])
            validation_loss, validation_accuracy = run_epoch(network, validation_loader, device)
            improved = validation_loss < best_loss
            if improved:
                best_loss, best_epoch = validation_loss, epoch
            stop = stopper.update(validation_loss)
            row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy,
                       validation_loss=validation_loss, validation_accuracy=validation_accuracy,
                       learning_rate=settings['learning_rate'], epoch_seconds=time.perf_counter()-epoch_start,
                       best_epoch=best_epoch, early_stop_stale=stopper.stale)
            rows.append(row); writer.writerow(row); stream.flush()
            payload = dict(format_version=1, architecture=ARCHITECTURE, network_config=network_config,
                           model_state_dict=network.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                           epoch=epoch, best_epoch=best_epoch, best_validation_loss=best_loss,
                           metrics=row, config=cfg, dataset_provenance=provenance,
                           channels=['obstacles','agent','goal'], actions=[[-1,0],[1,0],[0,-1],[0,1]],
                           environment=training.environment,
                           software={'torch':str(torch.__version__), 'numpy':str(np.__version__)},
                           rng_state={'torch':torch.get_rng_state(), 'data_loader':generator.get_state()},
                           selection_metric='minimum validation cross-entropy')
            if improved:
                atomic_checkpoint(output/'best.pt', payload)
            # final.pt is the latest fully completed epoch, including after an interrupted later epoch.
            atomic_checkpoint(output/'final.pt', payload)
            LOG.info('epoch=%d/%d train_loss=%.5f train_acc=%.3f val_loss=%.5f val_acc=%.3f best_epoch=%d',
                     epoch, settings['epochs'], train_loss, train_accuracy, validation_loss, validation_accuracy, best_epoch)
            if stop:
                LOG.info('Early stopping: %d epochs without a min_delta improvement', settings['patience'])
                break
    plot_metrics(rows, output/'training.png')
    summary = dict(epochs_completed=len(rows), best_epoch=best_epoch, best_validation_loss=best_loss,
                   final_validation_loss=rows[-1]['validation_loss'],
                   final_validation_accuracy=rows[-1]['validation_accuracy'],
                   stopped_early=len(rows)<settings['epochs'], duration_seconds=time.perf_counter()-started,
                   device=str(device), dataset_provenance=provenance,
                   note='Action accuracy is not navigation success. Run imitation.evaluate_bc separately.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n', encoding='utf8')
    LOG.info('Saved BC artifacts in %s; run closed-loop validation separately', output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/behavior_cloning.yaml')
    parser.add_argument('--device', choices=['auto','cpu','cuda'])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--output-directory', help='New, nonexistent run directory')
    args = parser.parse_args()
    with open(args.config, encoding='utf8') as stream:
        cfg = yaml.safe_load(stream)
    for argument, key in ((args.device,'device'), (args.epochs,'epochs'), (args.output_directory,'output_directory')):
        if argument is not None:
            cfg['training'][key] = argument
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    print(json.dumps(train(cfg), indent=2))


if __name__ == '__main__':
    main()
