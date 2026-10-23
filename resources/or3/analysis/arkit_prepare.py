"""Data-only ARKitScenes transformation and fixed matched-support protocol."""
from pathlib import Path
import sys,zipfile,json,csv,io,hashlib,time,argparse
import numpy as np
from PIL import Image,__version__ as PIL_VERSION
ROOT=Path(__file__).resolve().parent/'new_scene_validation'
SOURCE=Path(r'${DIODE_FREEZE_ROOT}')
sys.path.insert(0,str(SOURCE));import diode_common as dc

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');q.replace(p)
def transform(a,resample):
 return np.asarray(Image.fromarray(a).resize((320,240),resample).crop((8,6,312,234))).copy()
def source_frames(record):
 archive=ROOT/'archives'/(record['video_id']+'.zip')
 receipt=dc.read(archive.with_suffix('.json'));assert sha(archive)==receipt['sha256']
 z=zipfile.ZipFile(archive)
 rgb={Path(n).name:n for n in z.namelist() if '/wide/' in n and n.endswith('.png')}
 depth={Path(n).name:n for n in z.namelist() if '/highres_depth/' in n and n.endswith('.png')}
 names=sorted(set(rgb)&set(depth),key=lambda n:float(Path(n).stem.rsplit('_',1)[1]))
 assert len(names)>0 and set(rgb)==set(depth)
 positions=np.unique(np.rint(np.linspace(0,len(names)-1,min(15,len(names)))).astype(int))
 return z,rgb,depth,names,positions
def convert(record,index):
 z,rs,ds,names,positions=source_frames(record)
 dest=ROOT/'converted';dest.mkdir(exist_ok=True)
 masks=dc.masks();rows=[];supports=[]
 for position in positions:
  name=names[position];rb=z.read(rs[name]);db=z.read(ds[name])
  im=Image.open(io.BytesIO(rb)).convert('RGB');native=np.asarray(Image.open(io.BytesIO(db)))
  assert im.size==(1920,1440) and native.shape==(1440,1920) and native.dtype==np.uint16
  if record['sky_direction']=='Down':im=im.transpose(Image.Transpose.ROTATE_180);native=np.rot90(native,2).copy()
  assert record['sky_direction'] in ('Up','Down')
  rgb=np.asarray(im.resize((320,240),Image.Resampling.BILINEAR).crop((8,6,312,234)),dtype=np.uint8).copy()
  native= native.astype(np.float32)/1000.
  depth=transform(native,Image.Resampling.NEAREST)
  valid=np.isfinite(depth)&(depth>.1)&(depth<=10)
  depth[~valid]=0
  counts={law:int((valid&masks[law]).sum()) for law in dc.LAWS}
  primary=all(n>=500 for n in counts.values())
  ids=np.full((5,11,500),-1,dtype='<i4')
  for li,law in enumerate(dc.LAWS):
   candidates=np.flatnonzero(valid&masks[law])
   if len(candidates)<500:continue
   for rep,seed in enumerate(dc.SEEDS):
    sample=np.random.default_rng(seed+index).choice(candidates,500,replace=False).astype('<i4')
    ids[rep,li]=sample
    supports.append(dict(index=index,law=law,repetition=rep,seed=seed+index,coordinate_hash=hashlib.sha256(sample.tobytes()).hexdigest()))
  converted=dest/f'{index:04d}.npz'
  np.savez_compressed(converted,rgb=rgb,depth=depth,valid=valid,support_indices=ids)
  pairhash=hashlib.sha256(rgb.tobytes()+depth.tobytes()+valid.tobytes()).hexdigest()
  row=dict(index=index,scene='visit_'+record['visit_id'],scan='video_'+record['video_id'],
   visit_id=record['visit_id'],video_id=record['video_id'],sky_direction=record['sky_direction'],
   role=record['role'],image=name,temporal_position=int(position),available_frames=len(names),primary=primary,
   eligible_counts=counts,score_pixels=int(valid.sum()),score_fraction=float(valid.mean()),
   mean_depth_m=float(depth[valid].mean()) if valid.any() else None,
   raw_rgb_sha256=hashlib.sha256(rb).hexdigest(),raw_depth_sha256=hashlib.sha256(db).hexdigest(),
   archive_sha256=sha(ROOT/'archives'/(record['video_id']+'.zip')),
   converted=str(converted),sha256=sha(converted),pair_hash=pairhash)
  rows.append(row);index+=1
 z.close();return rows,supports,index

def run(dev=False):
 selection=dc.read(ROOT/'selection.json')['selected']
 selection=[r for r in selection if r['role']==('preprocessing_check' if dev else 'evaluation')]
 rows=[];supports=[];index=10000 if dev else 0
 for r in selection:
  rr,ss,index=convert(r,index);rows+=rr;supports+=ss
  print(json.dumps(dict(visit=r['visit_id'],frames=len(rr),eligible=sum(x['primary'] for x in rr))),flush=True)
 assert len({r['pair_hash'] for r in rows})==len(rows)
 name='development' if dev else 'evaluation'
 save(ROOT/(name+'_manifest.json'),dict(rows=rows,scope='data-only before predictions',source_selection_sha256=sha(ROOT/'selection.json')))
 p=ROOT/(name+'_supports.csv')
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(supports[0]));w.writeheader();w.writerows(supports)
 if dev:
  r=rows[len(rows)//2]
  with np.load(r['converted'],allow_pickle=False) as a:
   Image.fromarray(a['rgb']).save(ROOT/'preprocessing_rgb.png')
   np.save(ROOT/'preprocessing_depth.npy',a['depth'])
  print('DEV',len(rows),'eligible',sum(r['primary'] for r in rows),'manifest saved',flush=True)
  return
 primary=[r for r in rows if r['primary']]
 visits=set(r['scene'] for r in primary)
 assert len(visits)==20,'A selected venue has no eligible frames: resolve data-only exclusion before inference'
 reg=dc.read(SOURCE/'model_registry.json')
 codes=[Path(__file__),Path(__file__).with_name('diode_mechanism.py'),Path(__file__).with_name('arkit_evaluate.py'),
        SOURCE/'diode_common.py',dc.W/'completionformer_factorial_core.py',SOURCE.parent/'local_gate_screen_20261016/run_gate_screen.py']
 for p in codes:assert p.exists()
 frozen=dict(status='NEW_TARGET_PROTOCOL_FROZEN_BEFORE_TARGET_PREDICTIONS',created_unix=time.time(),
   target='ARKitScenes upsampling official Validation',scope='Frozen CompletionFormer Parent and source-trained gates; new project-local venue cohort',
   source_selection_sha256=sha(ROOT/'selection.json'),manifest_sha256=sha(ROOT/'evaluation_manifest.json'),
   support_sha256=sha(ROOT/'evaluation_supports.csv'),registry_sha256=sha(SOURCE/'model_registry.json'),
   preprocessing='Official native orthographic metric GT (uint16 mm to m); sky Up unchanged, Down rotate180 both RGB/GT; bilinear RGB and nearest depth to320x240 then crop228x304. GT finite and0.1<d<=10; invalid0, no inpainting.',
   candidate_population='Validation visits with at least one Up/Down video; select first lexicographic eligible video per visit; 20 evaluation visits, separate1 development visit; see immutable metadata selection',
   image_selection='Up to15 temporally equally spaced paired frames per video, rounded linspace, before GT eligibility; retain intersection all11 layouts >=500 points, no prediction-dependent selection; report all excluded',
   population_images=len(primary),candidate_images=len(rows),venues=20,
   point_count=500,laws=list(dc.LAWS),draw_seeds=list(dc.SEEDS),repetitions=5,
   primary_family=list(dc.LAWS[3:]),secondary_families={'uniform':['uniform'],'seen2':['left','right']},
   models=['parent']+[f'{a}_seed{s}' for a in ['spatial','image_mean'] for s in [42,43,44]],
   checkpoint_sha256={'parent':reg['parent']['checkpoint_sha256'],**{f"{r['arm']}_seed{r['seed']}":r['sha256'] for r in reg['models'] if r['arm'] in ('spatial','image_mean')}},
   code_hashes={str(p):sha(p) for p in codes},expected_score_rows=len(primary)*55*7,
   precision='FP32; audited CompletionFormer NLSPN6; no target training, calibration, threshold adjustment or depth scale fitting',
   outcomes=list(dc.METRICS),primary_comparison='Spatial versus Image Mean on venue-equal mean per-image RMSE in unseen8',
   aggregation='draw/layout mean within image; seed metrics separately and mean; images mean within visit; visits equal. Also image and video equal descriptive effects; one video/visit.',
   region_analysis='unseen8 regions as prior DIODE diagnosis; input separate, near<=1,middle1-2,far>2 in normalized support distance; additive MSE contributions and 2ec+c². Test far-region benefit and near/middle cost by visit, report counterexamples.',
   uncertainty='Report all20 venue paired effects, seed paired effects/sampleSD, win counts. No independent300-image or60-venue claims; no seed×venue pseudoreplication.',
   prior_exposure='Limited scoped local history search found upstream ARKit data loaders/README and a candidate note, no project-local predictions/data among searched roots; cannot prove all upstream pretraining histories. Published Parent labeled NYUv2; v1 local source adaptation NYUv2. Evaluating CompletionFormer only in this new cohort.',
   scientific_context='Protocol fixed after seeing DIODE aggregate and mechanism observations, before any ARKit evaluation predictions; explicit prospective replication of those observations on a new cohort',
   software={'numpy':np.__version__,'Pillow':PIL_VERSION},target_predictions_started=False)
 p=ROOT/'evaluation_plan_frozen.json';assert not p.exists(),'Do not overwrite frozen protocol'
 save(p,frozen);print('FROZEN',len(primary),'images',len(visits),'venues',frozen['expected_score_rows'],'score rows',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dev',action='store_true');a=p.parse_args();run(a.dev)
