"""V17 单卡模型 batch 显存与吞吐压测（模型计算，不含数据 IO）。"""
import argparse, json, time
import torch
from src.task.InteractionDynamics.residual_interaction_diffusion import ResidualInteractionDiffusion

p=argparse.ArgumentParser(description=__doc__); p.add_argument("--batches",nargs="+",type=int,default=[8,16,32,48,64,96,128]); p.add_argument("--warmup",type=int,default=10); p.add_argument("--steps",type=int,default=50); a=p.parse_args()
device=torch.device("cuda")
for batch in a.batches:
 try:
  torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); model=ResidualInteractionDiffusion().to(device); opt=torch.optim.AdamW(model.parameters(),2e-4)
  values=[torch.randn(batch,128,28,device=device),torch.randn(batch,128,4,device=device),torch.randn(batch,128,3,device=device),torch.randn(batch,128,32,6,device=device),torch.randn(batch,25,device=device),torch.randint(100,(batch,),device=device)]
  begin=time.perf_counter(); forward=backward=0
  for step in range(a.warmup+a.steps):
   torch.cuda.synchronize(); t=time.perf_counter()
   with torch.autocast("cuda",dtype=torch.bfloat16): out=model(*values); loss=out.square().mean()
   torch.cuda.synchronize(); f=time.perf_counter(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); torch.cuda.synchronize(); e=time.perf_counter()
   if step>=a.warmup: forward+=f-t; backward+=e-f
  elapsed=time.perf_counter()-begin
  print(json.dumps({"per_gpu_batch":batch,"global_batch":4*batch,"max_allocated_gb":torch.cuda.max_memory_allocated()/2**30,"max_reserved_gb":torch.cuda.max_memory_reserved()/2**30,"forward_ms":forward/a.steps*1000,"backward_ms":backward/a.steps*1000,"samples_per_second":batch*a.steps/(forward+backward),"oom":False}),flush=True)
  del model,opt,values,out,loss
 except torch.cuda.OutOfMemoryError:
  print(json.dumps({"per_gpu_batch":batch,"global_batch":4*batch,"oom":True}),flush=True); torch.cuda.empty_cache()
