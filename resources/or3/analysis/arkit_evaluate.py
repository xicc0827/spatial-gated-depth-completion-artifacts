"""Frozen new-venue validation: source-trained Parent/Spatial/ImageMean, no updates."""
from pathlib import Path
import sys,os,csv,json,time,argparse,hashlib
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from diode_mechanism import build,arr
ROOT=Path(__file__).resolve().parent/'new_scene_validation'
SOURCE=Path(r'${DIODE_FREEZE_ROOT}')
sys.path.insert(0,str(SOURCE));import diode_common as dc
dc.dependencies();import completionformer_factorial_core as cc
from gpu_serial_lease import gpu_lease

def save(p,x):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');q.replace(p)
def write(p,rows):
 temp=p.with_suffix('.tmp')
 with temp.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 temp.replace(p)
def run(dev=False):
 torch.set_num_threads(2)
 if dev:
  rows=[r for r in dc.read(ROOT/'development_manifest.json')['rows'] if r['primary']]
  rows=[rows[len(rows)//2]];plan=None;ph='engineering_development_only'
 else:
  plan=dc.read(ROOT/'evaluation_plan_frozen.json');ph=dc.sha(ROOT/'evaluation_plan_frozen.json')
  for p,h in plan['code_hashes'].items():assert dc.sha(p)==h,p
  assert dc.sha(ROOT/'evaluation_manifest.json')==plan['manifest_sha256']
  assert dc.sha(ROOT/'evaluation_supports.csv')==plan['support_sha256']
  assert dc.sha(SOURCE/'model_registry.json')==plan['registry_sha256']
  rows=[r for r in dc.read(ROOT/'evaluation_manifest.json')['rows'] if r['primary']]
  assert len(rows)==plan['population_images']
 dest=ROOT/('development_replay' if dev else 'per_image');dest.mkdir(exist_ok=True)
 supports={}
 with (ROOT/('development_supports.csv' if dev else 'evaluation_supports.csv')).open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):supports[(int(r['index']),r['law'],int(r['repetition']))]=r['coordinate_hash']
 completed=0;start=time.monotonic()
 with gpu_lease(),torch.inference_mode():
  base,correctors=build()
  environment=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),
   tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32)
  if not dev:save(ROOT/'evaluation_started.json',dict(plan_sha256=ph,unix=time.time(),pid=os.getpid(),**environment))
  for row in rows:
   p=dest/f"{row['index']:04d}.csv";rp=p.with_suffix('.regions.csv');marker=p.with_suffix('.json')
   if marker.exists():
    m=dc.read(marker);assert m['plan_sha256']==ph and m['sha256']==dc.sha(p) and m['region_sha256']==dc.sha(rp)
    completed+=1;continue
   assert dc.sha(row['converted'])==row['sha256']
   with np.load(row['converted'],allow_pickle=False) as a:rgb,gt,valid,indices=[a[k].copy() for k in ['rgb','depth','valid','support_indices']]
   image=torch.from_numpy(rgb).permute(2,0,1)[None].float().cuda()/255.
   scores=[];regions=[];identity_max=0.
   laws=['top'] if dev else list(dc.LAWS)
   for li,law in enumerate(dc.LAWS):
    if law not in laws:continue
    for rep in ([0] if dev else range(5)):
     ids=indices[rep,li];coordinate=hashlib.sha256(ids.tobytes()).hexdigest()
     assert coordinate==supports[(row['index'],law,rep)]
     sparse=np.zeros_like(gt);sparse.flat[ids]=gt.flat[ids]
     dep=torch.from_numpy(sparse)[None,None].cuda();view=cc.encode(base,image,dep)
     predictions={'parent':arr(cc.propagate(base,view,view['init']))}
     for name,c in correctors.items():predictions[name]=arr(cc.propagate(base,view,cc.corrected(view,cc.predict_delta(c,view))))
     for name,pred in predictions.items():
      metrics=dc.score(pred,gt,valid,ids)
      scores.append(dict(model=name,index=row['index'],scene=row['scene'],scan=row['scan'],image=row['image'],
       law=law,repetition=rep,coordinate_hash=coordinate,**metrics))
     if law not in dc.LAWS[3:]:continue
     observed=sparse>0;distance=distance_transform_edt(~observed)/np.sqrt(observed.size/500)
     masks={'input':valid&observed,'near_unobserved':valid&~observed&(distance<=1),
      'middle':valid&(distance>1)&(distance<=2),'far':valid&(distance>2),'all':valid}
     assert sum(int(m.sum()) for k,m in masks.items() if k!='all')==int(valid.sum())
     truth=gt.astype(np.float64)
     for seed in [42,43,44]:
      spatial=predictions[f'spatial_seed{seed}'].astype(np.float64)
      for control in ['image_mean','parent']:
       name=f'image_mean_seed{seed}' if control=='image_mean' else 'parent'
       baseline=predictions[name].astype(np.float64);e=baseline-truth;c=spatial-baseline;es=spatial-truth
       for region,m in masks.items():
        a=float(np.square(es[m]).sum());b=float(np.square(e[m]).sum());cross=float((2*e[m]*c[m]).sum());energy=float(np.square(c[m]).sum())
        identity_max=max(identity_max,abs(a-b-cross-energy));assert np.isclose(a-b,cross+energy,rtol=1e-10,atol=1e-7)
        regions.append(dict(index=row['index'],scene=row['scene'],scan=row['scan'],image=row['image'],law=law,repetition=rep,
         seed=seed,control=control,region=region,pixels=int(m.sum()),valid_pixels=int(valid.sum()),
         spatial_sse=a,control_sse=b,delta_sse=a-b,cross_term=cross,correction_energy=energy,
         opposing_pixels=int((e[m]*c[m]<0).sum()),benefit_pixels=int((es[m]**2<e[m]**2).sum())))
   assert len(scores)==(7 if dev else 385) and len(regions)==(30 if dev else 1200)
   write(p,scores);write(rp,regions)
   save(marker,dict(plan_sha256=ph,sha256=dc.sha(p),region_sha256=dc.sha(rp),score_rows=len(scores),region_rows=len(regions),identity_max=identity_max))
   completed+=1;status=dict(status='RUNNING',completed=completed,total=len(rows),last_index=row['index'],elapsed_seconds=time.monotonic()-start,pid=os.getpid(),plan_sha256=ph)
   save(ROOT/('development_status.json' if dev else 'status.json'),status);print(json.dumps(status),flush=True)
 if dev:save(ROOT/'preflight.json',dict(status='NEW_PREPROCESSING_MODEL_PATH_FINITE_OUTPUT_PASS',scope='engineering check only, separate development visit; no evaluation results',environment=environment,completed_images=completed))
 else:
  assert completed==len(rows)
  save(ROOT/'inference_complete.json',dict(status='ALL_FIXED_MODELS_NEW_VENUE_INFERENCE_COMPLETE',plan_sha256=ph,images=completed,
   score_rows=completed*385,region_rows=completed*1200,environment=environment,files={str(p):dc.sha(p) for p in dest.glob('*.csv')}))
  save(ROOT/'status.json',dict(status='COMPLETE',completed=completed,total=len(rows),elapsed_seconds=time.monotonic()-start,plan_sha256=ph))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dev',action='store_true');a=p.parse_args();run(a.dev)
