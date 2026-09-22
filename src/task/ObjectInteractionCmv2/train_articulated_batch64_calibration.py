"""One-step B64 mixed-domain calibration before the V1.8 formal run."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from .articulated import ArticulatedObjectInteractionCmv2V15Model, articulated_v15_loss
from .train_articulated_formal import _batch, _datasets, _move


def _write(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def _load(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if config.get('schema_name') != 'object_interaction_cmv2_two_domain_articulated_v1_8_batch64_calibration':
        raise ValueError('invalid V1.8 B64 calibration config')
    for key in ('source_index', 'source_manifest'):
        config[key] = str(Path(config[key]).resolve())
    config['articulation_metadata'] = str((path.resolve().parents[5] / config['articulation_metadata']).resolve())
    if any(not Path(config[key]).is_file() for key in ('source_index', 'source_manifest', 'articulation_metadata')):
        raise FileNotFoundError('V1.8 calibration input missing')
    training = config['training']
    if training['device'] != 'cuda:1' or training['batch_size'] != 64 or training['smoke_steps'] != 1:
        raise ValueError('unapproved V1.8 B64 calibration budget')
    if config['data']['train_stride_values'] != {'grab': [1], 'arctic': [5, 6, 7, 8, 9, 10]}:
        raise ValueError('V1.8 calibration stride contract mismatch')
    return config


def run(config: dict, run_id: str) -> dict:
    device = torch.device(config['training']['device'])
    torch.cuda.set_device(device)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info(device)[0] < config['training']['minimum_free_memory_gib'] * 2**30:
        raise RuntimeError('GPU1 unavailable for approved B64 calibration')
    torch.manual_seed(config['training']['seed'])
    torch.cuda.manual_seed_all(config['training']['seed'])
    output = Path(config['output_root']) / run_id
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'config.json', config)
    manifest = {
        'schema_name': 'ref2dex_run_manifest_v1', 'task': 'ObjectInteractionCmv2',
        'operation': 'v1_8_v15_random_init_batch64_calibration', 'run_id': run_id,
        'run_status': 'STARTED', 'work_version': config['work_version'],
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'architecture_version': 'v1_5_articulated_fk', 'initial_checkpoint': None,
        'config': 'config.json', 'outputs': {'metrics': 'metrics.jsonl', 'train_log': 'train.log', 'latest_checkpoint': 'latest.pt'},
        'conclusion': 'INCONCLUSIVE',
    }
    _write(output / 'run_manifest.json', manifest)
    try:
        datasets = _datasets(config, 'train')
        generator = torch.Generator().manual_seed(config['training']['seed'])
        model = ArticulatedObjectInteractionCmv2V15Model(SimpleNamespace(**config['model'])).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        raw = _batch(datasets, config['training']['batch_size'], generator)
        batch = _move(raw, device)
        optimizer.zero_grad(set_to_none=True)
        losses = articulated_v15_loss(model(batch), batch)
        if not torch.isfinite(losses['total']):
            raise ValueError('non-finite B64 calibration loss')
        losses['total'].backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        record = {
            'phase': 'smoke', 'step': 1, 'batch_size': 64,
            'loss': float(losses['total'].detach().cpu()), 'flow_loss': float(losses['flow'].detach().cpu()),
            'source_counts': {'grab': 32, 'arctic': 32},
        }
        line = json.dumps(record)
        (output / 'metrics.jsonl').write_text(line + '\n')
        (output / 'train.log').write_text(line + '\n')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'architecture_version': model.architecture_version,
                    'work_version': config['work_version'], 'step': 1, 'seed': config['training']['seed'],
                    'calibrated_batch_size': 64}, output / 'latest.pt')
        manifest.update({'run_status': 'COMPLETED', 'last_step': 1, 'last_epoch': 1, 'best_metric': None,
                         'calibrated_batch_size': 64, 'source_counts': record['source_counts'], 'conclusion': 'SUPPORTED'})
        _write(output / 'run_manifest.json', manifest)
        return manifest
    except Exception as error:
        manifest.update({'run_status': 'FAILED', 'error': repr(error), 'conclusion': 'INVALID_IMPLEMENTATION'})
        _write(output / 'run_manifest.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(_load(arguments.config), arguments.run_id), indent=2))
