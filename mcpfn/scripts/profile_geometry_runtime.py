"""Stage timings and inference-only chunk-size experiments; leaves model source unchanged."""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import numpy as np
import torch
from tabimpute.interface import ImputePFN
from benchmark_dwip_geometry_runtime import DEFAULT_CHECKPOINT


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output-dir',type=Path,default=REPO/'runs/benchmark_evaluations/geometry_runtime_profile')
 p.add_argument('--sizes',type=int,nargs='+',default=[500,1000])
 p.add_argument('--chunks',type=int,nargs='+',default=[128,512,2048])
 args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
 if (args.output_dir/'results.json').exists(): raise FileExistsError('Use a fresh output directory')
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8)
 cfg=json.loads((DEFAULT_CHECKPOINT.parent/'config.json').read_text())
 imp=ImputePFN(device='cuda',checkpoint_path=str(DEFAULT_CHECKPOINT),entry_wise_features=False,json_config={'config':cfg['model']},model_family=cfg['model_family'],precision_mode='amp_bfloat16',prediction_stat='mean',max_num_rows=None,max_num_cols=None)
 model=imp.model
 state={'enabled':False,'events':[],'stack':[]}
 def instrument(obj,attr,label):
  original=getattr(obj,attr)
  def wrapped(*a,**kw):
   if not state['enabled']: return original(*a,**kw)
   path='/'.join(state['stack']+[label]);state['stack'].append(label)
   begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   begin.record();t=time.perf_counter()
   try: return original(*a,**kw)
   finally:
    end.record();state['events'].append((path,begin,end,time.perf_counter()-t));state['stack'].pop()
  setattr(obj,attr,wrapped)
 for attr,label in [('encode_latents','encoder'),('_point_grid','solver_total'),('_svt_iterates','svt_iterates'),('_svt_mixture_cv_rmse','cv'),('_decode_with_svt_features','decoder_total'),('_observed_neighbor_point_features','neighbor_features'),('_observed_attention_context','observed_attention'),('_observed_attention_bias','value_bias')]: instrument(model,attr,label)
 instrument(model.cell_query_decoder,'forward','cell_query')
 instrument(model.output_decoder,'decode_features','distribution_head')
 instrument(imp,'_point_predictions','distribution_mean')
 records=[]
 for n in args.sizes:
  rng=np.random.default_rng(np.random.SeedSequence([20260915,n]));rank=min(10,n)
  x=rng.standard_normal((n,rank))@rng.standard_normal((rank,n))/rank**.5
  mask=np.zeros(n*n,bool);mask[rng.choice(n*n,round(n*n*.8),replace=False)]=True;mask=mask.reshape(n,n);x[mask]=np.nan
  reference=None
  for variant,chunk,query in [(f'chunk_{c}',c,True) for c in args.chunks]+[('no_query_diagnostic',128,False)]:
   model.cell_query_decoder.chunk_size=chunk;model.use_cell_query_decoder=query
   torch.cuda.empty_cache()
   record=dict(size=n,variant=variant,chunk=chunk,query_enabled=query,query_chunks=int(np.ceil(mask.sum()/chunk)) if query else 0)
   print(f'Start {n} {variant}',flush=True)
   try:
    pred=imp.impute(x.copy());del pred;torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats();times=[];stages={}
    for rep in range(2):
     inp=x.copy();state['events']=[];state['enabled']=rep==0
     torch.cuda.synchronize();start=time.perf_counter()
     pred=imp.impute(inp);torch.cuda.synchronize();times.append(time.perf_counter()-start)
     state['enabled']=False
     if rep==0:
      grouped=defaultdict(lambda:dict(calls=0,gpu_seconds=0.,cpu_enqueue_seconds=0.))
      for path,begin,end,cpu in state['events']:
       v=grouped[path];v['calls']+=1;v['gpu_seconds']+=begin.elapsed_time(end)/1000;v['cpu_enqueue_seconds']+=cpu
      stages=dict(grouped)
    assert np.isfinite(pred).all()
    if reference is None: reference=pred.copy()
    delta=pred[mask]-reference[mask]
    record.update(status='ok',seconds=times,unprofiled_seconds=times[1],stages=stages,max_abs_prediction_change=float(np.max(np.abs(delta))),rms_prediction_change=float(np.sqrt(np.mean(delta**2))),peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
    del pred
   except torch.cuda.OutOfMemoryError as error:
    state['enabled']=False;state['stack']=[];state['events']=[]
    record.update(status='out_of_memory',error=str(error));torch.cuda.empty_cache()
   records.append(record)
   (args.output_dir/'results.json').write_text(json.dumps(records,indent=2))
   print(json.dumps(record),flush=True)
 print('Completed',flush=True)

if __name__=='__main__': main()
