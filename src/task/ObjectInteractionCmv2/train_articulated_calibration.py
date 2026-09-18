"""Bounded V1.6 calibration for randomly initialized V1.5 articulated Cmv2."""
from __future__ import annotations

import argparse, hashlib, json, subprocess, time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from .articulated import ArticulatedObjectInteractionCmv2V15Model, ArticulatedTransitions, articulated_v15_loss, collate_articulated


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def _load(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if cfg.get('schema_name') != 'object_interaction_cmv2_two_domain_articulated_v1_6_calibration':
        raise ValueError('invalid V1.6 calibration config')
    for key in ('source_index', 'source_manifest', 'articulation_metadata'):
        cfg[key] = str((path.parents[5] / cfg[key]).resolve()) if key == 'articulation_metadata' else str(Path(cfg[key]).resolve())
        if not Path(cfg[key]).is_file(): raise FileNotFoundError(cfg[key])
    if cfg['training']['device'] != 'cuda:1' or cfg['training']['batch_candidates'] != [2,4,8,16] or cfg['training']['smoke_steps'] != 8:
        raise ValueError('unapproved V1.6 calibration budget')
    return cfg


def _datasets(cfg: dict):
    index = json.loads(Path(cfg['source_index']).read_text())['sequences']['train']
    metadata = json.loads(Path(cfg['articulation_metadata']).read_text())['articulation']
    result = []
    for domain in ('grab', 'arctic'):
        entry = next(item for item in index if item.get('dataset') == domain and item.get('hand_variant', item.get('variant')) == 'mano')
        articulation = {'num_links': 1, 'joints': []} if domain == 'grab' else metadata
        result.append(ArticulatedTransitions([{'name': domain, 'path': entry['path'], 'hand_variant': 'mano', 'articulation': articulation, 'id': entry.get('id')}], 'train', num_obj_points=cfg['data']['num_obj_points'], fixed_stride=1, base_seed=cfg['training']['seed']))
    return result


def _batch(datasets, batch_size: int, step: int):
    samples=[]
    for position in range(batch_size):
        ds=datasets[position % 2]
        samples.append(ds[(step * batch_size + position) % len(ds)])
    return collate_articulated(samples)


def _move(batch, device): return {k: v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}


def run(cfg: dict, run_id: str) -> dict:
    device=torch.device(cfg['training']['device'])
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.cuda.set_device(device)
    if torch.cuda.mem_get_info(device)[0] < cfg['training']['minimum_free_memory_gib'] * 2**30: raise RuntimeError('GPU1 free memory below approved minimum')
    torch.manual_seed(cfg['training']['seed']); torch.cuda.manual_seed_all(cfg['training']['seed'])
    output=Path(cfg['output_root'])/run_id; output.mkdir(parents=True, exist_ok=False)
    _write(output/'config.json', cfg)
    manifest={'schema_name':'ref2dex_run_manifest_v1','task':'ObjectInteractionCmv2','operation':'v1_6_v15_random_init_batch_calibration','run_id':run_id,'run_status':'STARTED','modification_version':cfg['modification_version'],'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'architecture_version':'v1_5_articulated_fk','initial_checkpoint':None,'config':'config.json','input':{'index':cfg['source_index'],'index_sha256':_sha(Path(cfg['source_index'])),'manifest':cfg['source_manifest'],'manifest_sha256':_sha(Path(cfg['source_manifest'])),'articulation_metadata':cfg['articulation_metadata'],'articulation_metadata_sha256':_sha(Path(cfg['articulation_metadata']))},'outputs':{'metrics':'metrics.jsonl','train_log':'train.log','latest_checkpoint':'latest.pt'},'conclusion':'INCONCLUSIVE'}
    _write(output/'run_manifest.json',manifest)
    try:
        datasets=_datasets(cfg); successes=[]
        with (output/'metrics.jsonl').open('w') as metrics, (output/'train.log').open('w') as log:
            for candidate in cfg['training']['batch_candidates']:
                try:
                    model=ArticulatedObjectInteractionCmv2V15Model(SimpleNamespace(**cfg['model'])).to(device); opt=torch.optim.Adam(model.parameters(),lr=cfg['training']['learning_rate'])
                    raw=_batch(datasets,candidate,0); batch=_move(raw,device); opt.zero_grad(set_to_none=True); loss=articulated_v15_loss(model(batch),batch)['total']
                    if not torch.isfinite(loss): raise ValueError('non-finite calibration loss')
                    loss.backward(); opt.step(); torch.cuda.synchronize(device); successes.append(candidate)
                    record={'phase':'calibration','batch_size':candidate,'loss':float(loss.detach().cpu()),'status':'OK'}
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); record={'phase':'calibration','batch_size':candidate,'status':'OOM'}
                line=json.dumps(record); metrics.write(line+'\n'); log.write(line+'\n'); metrics.flush(); log.flush()
            if not successes: raise RuntimeError('all approved batch candidates OOM')
            selected=max(successes); model=ArticulatedObjectInteractionCmv2V15Model(SimpleNamespace(**cfg['model'])).to(device); opt=torch.optim.Adam(model.parameters(),lr=cfg['training']['learning_rate'])
            for step in range(1,cfg['training']['smoke_steps']+1):
                raw=_batch(datasets,selected,step); batch=_move(raw,device); opt.zero_grad(set_to_none=True); out=model(batch); losses=articulated_v15_loss(out,batch)
                if not torch.isfinite(losses['total']): raise ValueError(f'non-finite loss at {step}')
                losses['total'].backward(); opt.step()
                record={'phase':'smoke','step':step,'batch_size':selected,'loss':float(losses['total'].detach().cpu()),'flow_loss':float(losses['flow'].detach().cpu()),'source_counts':{'grab':selected//2,'arctic':selected//2}}
                line=json.dumps(record); metrics.write(line+'\n'); log.write(line+'\n'); metrics.flush(); log.flush()
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'architecture_version':model.architecture_version,'modification_version':cfg['modification_version'],'step':cfg['training']['smoke_steps'],'seed':cfg['training']['seed'],'calibrated_batch_size':selected},output/'latest.pt')
        manifest.update({'run_status':'COMPLETED','last_step':cfg['training']['smoke_steps'],'last_epoch':1,'best_metric':None,'calibrated_batch_size':selected,'conclusion':'SUPPORTED'}); _write(output/'run_manifest.json',manifest); return manifest
    except Exception as error:
        manifest.update({'run_status':'FAILED','error':repr(error),'conclusion':'INVALID_IMPLEMENTATION'}); _write(output/'run_manifest.json',manifest); raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True); parser.add_argument('--run-id',required=True); args=parser.parse_args(); print(json.dumps(run(_load(args.config),args.run_id),indent=2))
