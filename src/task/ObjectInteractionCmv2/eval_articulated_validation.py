"""Full validation diagnostic for the frozen V1.5 articulated checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from .articulated import ArticulatedObjectInteractionCmv2V15Model, articulated_v15_loss, collate_articulated
from .eval_two_domain_mano import FlowMetrics
from .train_articulated_formal import _datasets, _move


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _write(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if config.get('schema_name') != 'object_interaction_cmv2_two_domain_articulated_v1_9_validation_eval':
        raise ValueError('invalid V1.9 validation-evaluation config')
    for key in ('checkpoint', 'source_index', 'source_manifest'):
        config[key] = str(Path(config[key]).resolve())
    config['articulation_metadata'] = str((path.resolve().parents[5] / config['articulation_metadata']).resolve())
    if any(not Path(config[key]).is_file() for key in ('checkpoint', 'source_index', 'source_manifest', 'articulation_metadata')):
        raise FileNotFoundError('V1.9 validation-evaluation input missing')
    if config.get('modification_version') != 'V1.9.2':
        raise ValueError('V1.9 modification version mismatch')
    if config['data']['train_stride_values'] != {'grab': [1], 'arctic': [5, 6, 7, 8, 9, 10]}:
        raise ValueError('V1.9 stride contract mismatch')
    evaluation = config['evaluation']
    if evaluation['device'] != 'cuda:1' or evaluation['batch_size'] != 64 or evaluation['seed'] != 42:
        raise ValueError('unapproved V1.9 validation-evaluation budget')
    return config


def _summary(metrics: FlowMetrics, loss_sum: float, samples: int) -> dict:
    return {**metrics.summary(), 'structured_loss_sample_mean': loss_sum / samples}


@torch.inference_mode()
def _evaluate_domain(model: ArticulatedObjectInteractionCmv2V15Model, dataset, domain: str, config: dict,
                     device: torch.device) -> tuple[dict, dict[str, FlowMetrics]]:
    epsilon = float(config['evaluation']['angle_epsilon_m'])
    total = FlowMetrics(epsilon)
    per_stride: dict[str, FlowMetrics] = {}
    loss_sum, samples = 0.0, 0
    batch_size = int(config['evaluation']['batch_size'])
    model.eval()
    for offset in range(0, len(dataset), batch_size):
        raw = collate_articulated([dataset[index] for index in range(offset, min(offset + batch_size, len(dataset)))])
        batch = _move(raw, device)
        output = model(batch)
        loss = articulated_v15_loss(output, batch)['total']
        count = int(batch['obj_points'].shape[0])
        prediction, target = output['obj_flow_pred'], batch['obj_flow_gt']
        total.update(prediction, target)
        loss_sum += float(loss.cpu()) * count
        samples += count
        for stride in torch.unique(batch['stride']).tolist():
            stride_key = str(int(stride))
            metric = per_stride.setdefault(stride_key, FlowMetrics(epsilon))
            mask = batch['stride'] == int(stride)
            metric.update(prediction[mask], target[mask])
    if samples != len(dataset) or total.samples != len(dataset):
        raise RuntimeError(f'{domain} sample accounting mismatch: {samples}/{total.samples}/{len(dataset)}')
    return _summary(total, loss_sum, samples), per_stride


def run(config: dict, run_id: str) -> dict:
    device = torch.device(config['evaluation']['device'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable for V1.9 evaluation')
    torch.cuda.set_device(device)
    if torch.cuda.mem_get_info(device)[0] < int(config['evaluation']['minimum_free_memory_gib']) * 2**30:
        raise RuntimeError('GPU1 free memory below V1.9 evaluation minimum')
    checkpoint_path = Path(config['checkpoint'])
    output = Path(config['output_root']) / run_id
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'config.json', config)
    manifest = {
        'schema_name': 'ref2dex_run_manifest_v1', 'task': 'ObjectInteractionCmv2',
        'operation': 'v1_9_v15_articulated_validation_evaluation', 'run_id': run_id,
        'run_status': 'STARTED', 'created_at': _utc_now(), 'modification_version': config['modification_version'],
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'checkpoint': str(checkpoint_path), 'checkpoint_sha256': _sha256(checkpoint_path),
        'initial_checkpoint': None, 'split': 'val', 'architecture_version': 'v1_5_articulated_fk',
        'stride_contract': {'grab': [1], 'arctic': [5, 6, 7, 8, 9, 10], 'assignment': 'stable_per_sequence_current_frame'},
        'outputs': {'metrics': 'metrics.jsonl', 'summary': 'metrics_summary.json', 'log': 'eval.log'},
        'conclusion': 'INCONCLUSIVE',
    }
    _write(output / 'run_manifest.json', manifest)
    try:
        payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model = ArticulatedObjectInteractionCmv2V15Model(SimpleNamespace(**config['model'])).to(device)
        if payload.get('architecture_version') != model.architecture_version or payload.get('modification_version') != 'V1.8.1':
            raise ValueError('checkpoint is not the frozen V1.8 articulated best checkpoint')
        model.load_state_dict(payload['model'], strict=True)
        dataset_config = {**config, 'training': {'seed': config['evaluation']['seed']}}
        datasets = _datasets(dataset_config, 'val')
        manifest.update({'run_status': 'RUNNING', 'started_at': _utc_now(),
                         'rows_by_domain': {key: len(value) for key, value in datasets.items()},
                         'checkpoint_epoch': payload.get('epoch'), 'checkpoint_step': payload.get('step')})
        _write(output / 'run_manifest.json', manifest)
        groups, per_stride = {}, {}
        for domain in ('grab', 'arctic'):
            groups[domain], grouped = _evaluate_domain(model, datasets[domain], domain, config, device)
            if domain == 'arctic':
                per_stride = {f'arctic_stride_{key}': value.summary() for key, value in sorted(grouped.items(), key=lambda item: int(item[0]))}
        summary = {'run_id': run_id, 'split': 'val', 'checkpoint': str(checkpoint_path), 'checkpoint_epoch': payload.get('epoch'),
                   'checkpoint_step': payload.get('step'), 'groups': groups, 'arctic_by_actual_stride': per_stride}
        _write(output / 'metrics_summary.json', summary)
        (output / 'metrics.jsonl').write_text(json.dumps(summary, ensure_ascii=False) + '\n')
        (output / 'eval.log').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        manifest.update({'run_status': 'COMPLETED', 'finished_at': _utc_now(),
                         'last_epoch': payload.get('epoch'), 'last_step': payload.get('step'),
                         'best_metric': payload.get('best_metric'), 'conclusion': 'INCONCLUSIVE'})
        _write(output / 'run_manifest.json', manifest)
        return summary
    except Exception as error:
        manifest.update({'run_status': 'FAILED', 'finished_at': _utc_now(), 'error': repr(error),
                         'conclusion': 'INVALID_IMPLEMENTATION'})
        _write(output / 'run_manifest.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(_load(arguments.config), arguments.run_id), ensure_ascii=False, indent=2))
