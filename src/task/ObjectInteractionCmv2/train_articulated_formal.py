"""V1.7 full, random-init two-domain articulated training."""
from __future__ import annotations
import argparse, hashlib, json, math, subprocess, time
from pathlib import Path
from types import SimpleNamespace
import torch, yaml
from .articulated import ArticulatedObjectInteractionCmv2V15Model, ArticulatedTransitions, articulated_v15_loss, collate_articulated

def _write(path, value):
    temp=path.with_name(path.name+'.tmp'); temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n'); temp.replace(path)
def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _load(path):
    cfg=yaml.safe_load(Path(path).read_text())
    expected={'object_interaction_cmv2_two_domain_articulated_v1_7_formal':('V1.7.1',16,{'grab':[1],'arctic':[5,6,7,8,9,10]}),'object_interaction_cmv2_two_domain_articulated_v1_8_formal':('V1.8.1',64,{'grab':[1],'arctic':[5,6,7,8,9,10]}),'object_interaction_cmv2_two_domain_articulated_v1_11_formal':('V1.11.1',64,{'grab':[1,2,3],'arctic':[1,2,3]})}
    if cfg.get('schema_name') not in expected: raise ValueError('invalid articulated formal config')
    for key in ('source_index','source_manifest'): cfg[key]=str(Path(cfg[key]).resolve())
    cfg['articulation_metadata']=str((Path(path).resolve().parents[5]/cfg['articulation_metadata']).resolve())
    if any(not Path(cfg[k]).is_file() for k in ('source_index','source_manifest','articulation_metadata')): raise FileNotFoundError('V1.7 input missing')
    version,batch_size,strides=expected[cfg['schema_name']]
    if cfg['work_version']!=version or cfg['training']['device']!='cuda:1' or cfg['training']['batch_size']!=batch_size or cfg['training']['epochs']!=16: raise ValueError('unapproved formal budget')
    if cfg['data']['train_stride_values']!=strides: raise ValueError('formal stride contract mismatch')
    return cfg
def _datasets(cfg, split, *, validation=False):
    entries=json.loads(Path(cfg['source_index']).read_text())['sequences'][split]; art=json.loads(Path(cfg['articulation_metadata']).read_text())['articulation']; out={}
    for domain in ('grab','arctic'):
        specs=[{'name':domain,'path':x['path'],'hand_variant':'mano','id':x.get('id'),'articulation':({'num_links':1,'joints':[]} if domain=='grab' else art)} for x in entries if x.get('dataset')==domain and x.get('hand_variant',x.get('variant'))=='mano']
        if not specs: raise ValueError(f'no {split}/{domain} specs')
        out[domain]=ArticulatedTransitions(specs,split,num_obj_points=cfg['data']['num_obj_points'],stride_values=([cfg['data']['eval_stride']] if validation else cfg['data']['train_stride_values'][domain]),base_seed=cfg['training']['seed'])
    return out
def _move(b,d): return {k:v.to(d) if torch.is_tensor(v) else v for k,v in b.items()}
def _batch(ds, size, gen):
    half=size//2; samples=[]
    for domain in ('grab','arctic'):
        picks=torch.randint(len(ds[domain]),(half,),generator=gen).tolist(); samples += [ds[domain][i] for i in picks]
    return collate_articulated(samples)
def _checkpoint(path, model,opt,cfg,step,epoch,best,meta): torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'architecture_version':model.architecture_version,'work_version':cfg['work_version'],'step':step,'epoch':epoch,'seed':cfg['training']['seed'],'best_metric':best,'metadata':meta},path)
@torch.inference_mode()
def _evaluate(model, datasets, cfg, device):
    result={}; model.eval()
    for domain,ds in datasets.items():
        total=count=0.;
        for offset in range(0,len(ds),cfg['training']['batch_size']):
            raw=collate_articulated([ds[i] for i in range(offset,min(offset+cfg['training']['batch_size'],len(ds)))])
            loss=articulated_v15_loss(model(_move(raw,device)),_move(raw,device))['total']; total+=float(loss.cpu())*(min(cfg['training']['batch_size'],len(ds)-offset)); count+=min(cfg['training']['batch_size'],len(ds)-offset)
        result[f'val_{domain}_loss']=total/count
    result['selection_metric']=.5*(result['val_grab_loss']+result['val_arctic_loss']); return result
def run(cfg,run_id):
    device=torch.device('cuda:1'); torch.cuda.set_device(device)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info(device)[0]<cfg['training']['minimum_free_memory_gib']*2**30: raise RuntimeError('GPU1 unavailable')
    torch.manual_seed(cfg['training']['seed']); torch.cuda.manual_seed_all(cfg['training']['seed']); out=Path(cfg['output_root'])/run_id; out.mkdir(parents=True,exist_ok=False); _write(out/'config.json',cfg)
    manifest={'schema_name':'ref2dex_run_manifest_v1','task':'ObjectInteractionCmv2','operation':f"{cfg['work_version'].lower().replace('.', '_')}_v15_random_init_formal_train",'run_id':run_id,'run_status':'STARTED','work_version':cfg['work_version'],'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'architecture_version':'v1_5_articulated_fk','initial_checkpoint':None,'config':'config.json','input':{k:cfg[k] for k in ('source_index','source_manifest','articulation_metadata')},'outputs':{'metrics':'metrics.jsonl','train_log':'train.log','latest_checkpoint':'latest.pt','best_checkpoint':'best.pt'},'conclusion':'INCONCLUSIVE'}; _write(out/'run_manifest.json',manifest)
    try:
      train=_datasets(cfg,'train'); val=_datasets(cfg,'val',validation=True); steps=math.ceil((len(train['grab'])+len(train['arctic']))/cfg['training']['batch_size']); model=ArticulatedObjectInteractionCmv2V15Model(SimpleNamespace(**cfg['model'])).to(device); opt=torch.optim.Adam(model.parameters(),lr=cfg['training']['learning_rate']); gen=torch.Generator().manual_seed(cfg['training']['seed']); best=None; step=0
      manifest.update({'run_status':'RUNNING','train_rows_by_source':{k:len(v) for k,v in train.items()},'val_rows_by_source':{k:len(v) for k,v in val.items()},'steps_per_epoch':steps});_write(out/'run_manifest.json',manifest)
      with (out/'metrics.jsonl').open('w') as mf,(out/'train.log').open('w') as lf:
       for epoch in range(1,cfg['training']['epochs']+1):
        model.train()
        for _ in range(steps):
         raw=_batch(train,cfg['training']['batch_size'],gen); batch=_move(raw,device); opt.zero_grad(set_to_none=True); losses=articulated_v15_loss(model(batch),batch)
         if not torch.isfinite(losses['total']): raise ValueError(f'non-finite step {step+1}')
         losses['total'].backward();opt.step();step+=1
         if step%cfg['training']['checkpoint_interval']==0:_checkpoint(out/'latest.pt',model,opt,cfg,step,epoch,best,{})
        _checkpoint(out/'latest.pt',model,opt,cfg,step,epoch,best,{}) ; values=_evaluate(model,val,cfg,device); improved=best is None or values['selection_metric']<best
        if improved: best=values['selection_metric'];_checkpoint(out/'best.pt',model,opt,cfg,step,epoch,best,{'validation':values})
        record={'epoch':epoch,'step':step,**values,'best_metric':best,'improved':improved};line=json.dumps(record);mf.write(line+'\n');lf.write(line+'\n');mf.flush();lf.flush()
      manifest.update({'run_status':'COMPLETED','last_step':step,'last_epoch':cfg['training']['epochs'],'best_metric':best,'conclusion':'INCONCLUSIVE'});_write(out/'run_manifest.json',manifest);return manifest
    except Exception as e: manifest.update({'run_status':'FAILED','error':repr(e),'conclusion':'INVALID_IMPLEMENTATION'});_write(out/'run_manifest.json',manifest);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();print(json.dumps(run(_load(a.config),a.run_id),indent=2))
