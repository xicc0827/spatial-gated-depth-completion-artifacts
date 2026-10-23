"""Post-hoc fixed-checkpoint DIODE mechanism diagnosis; no training."""
from pathlib import Path
import os, sys, json, csv, time, argparse, hashlib
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'mechanism'
SRC=Path(r'${DIODE_FREEZE_ROOT}')
sys.path.insert(0,str(SRC))
import diode_common as dc
dc.dependencies()
import completionformer_factorial_core as cc
import run_gate_screen as gc
from completionformer_port import build_uninitialized, load_pretrained
from gpu_serial_lease import gpu_lease

def save(p,x):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');q.replace(p)

def prepare():
    OUT.mkdir(exist_ok=True)
    plan=dc.read(SRC/'evaluation_plan_frozen.json')
    for p,h in plan['code_hashes'].items(): assert dc.sha(p)==h,p
    rows=[r for r in dc.read(SRC/'diode_converted_manifest.json')['rows'] if r['primary']]
    exemplars={}
    for r in sorted(rows,key=lambda r:r['index']): exemplars.setdefault(r['scan'],r['index'])
    config=dict(scope='POST_HOC existing exposed DIODE cohort; no training; frozen Parent and six gate checkpoints',
        source_plan_sha256=dc.sha(SRC/'evaluation_plan_frozen.json'),
        script_sha256=dc.sha(__file__),indices=[r['index'] for r in rows],laws=list(dc.LAWS[3:]),
        repetitions=list(range(5)),seeds=[42,43,44],
        regions='input; near_unobserved: 0<d/sqrt(HW/500)<=1; middle: 1<d/sqrt(HW/500)<=2; far: >2; valid mask unchanged',
        decomposition='e=ImageMean-GT,c=Spatial-ImageMean; delta SSE=2sum(e*c)+sum(c*c); also paired vs Parent',
        outputs='per-case regional SSE, direction and magnitude; final and internal pixel arrays for first image of each scan, first unseen layout, repetition0 only',
        exemplars=exemplars,created_unix=time.time())
    p=OUT/'plan.json'
    if p.exists():
        old=dc.read(p)
        for k in config:
            if k!='created_unix':assert old[k]==config[k],k
    else:save(p,config)
    return rows,config

def build():
    reg=dc.read(SRC/'model_registry.json');cf=dc.read(dc.W/'completionformer_factorial/protocol_frozen.json')
    assert dc.sha(reg['parent']['checkpoint'])==reg['parent']['checkpoint_sha256']
    for name,h in cf['port_sha256'].items():assert dc.sha(dc.W/'source_snapshot'/name)==h
    base=build_uninitialized(cf['upstream_source'],prop_time=6)
    load_pretrained(base,reg['parent']['checkpoint']);base=base.cuda().eval().requires_grad_(False)
    correctors={}
    for slot in reg['models']:
        if slot['arm'] not in ('spatial','image_mean'):continue
        assert dc.sha(slot['checkpoint'])==slot['sha256']
        s=torch.load(slot['checkpoint'],map_location='cpu',weights_only=False)
        assert s['applied']==2000 and s['parent_sha256']==reg['parent']['checkpoint_sha256']
        c=gc.gated_fresh('CompletionFormer',True,slot['arm'])
        c.load_state_dict(s['corrector'],strict=True)
        assert sum(p.numel() for p in c.parameters())==24513
        correctors[f"{slot['arm']}_seed{slot['seed']}"]=c.cuda().eval().requires_grad_(False)
    assert len(correctors)==6
    return base,correctors

def arr(x):return x[0,0].float().cpu().numpy()

def run(limit):
    rows,plan=prepare();ph=dc.sha(OUT/'plan.json')
    done_dir=OUT/'per_image';done_dir.mkdir(exist_ok=True)
    pixel_dir=OUT/'pixels';pixel_dir.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    start=time.monotonic();completed=0;maximum_replay=0.
    with gpu_lease(),torch.inference_mode():
        base,correctors=build()
        for row in rows:
            path=done_dir/f"{row['index']:04d}.csv";marker=path.with_suffix('.json')
            if marker.exists():
                m=dc.read(marker);assert m['plan_sha256']==ph and m['sha256']==dc.sha(path)
                completed+=1;continue
            assert dc.sha(row['converted'])==row['sha256']
            with np.load(row['converted'],allow_pickle=False) as a:
                rgb,gt,valid,indices=[a[k].copy() for k in ['rgb','depth','valid','support_indices']]
            ref={}
            with (SRC/'per_frame'/path.name).open(encoding='utf-8-sig',newline='') as f:
                for r in csv.DictReader(f):ref[(r['model'],r['law'],int(r['repetition']))]=r
            image=torch.from_numpy(rgb).permute(2,0,1)[None].float().cuda()/255.
            records=[];local_replay=0.;local_identity=0.;pixel_hashes={}
            for li,law in enumerate(dc.LAWS):
                if law not in plan['laws']:continue
                for rep in plan['repetitions']:
                    ids=indices[rep,li]
                    ch=hashlib.sha256(ids.tobytes()).hexdigest()
                    assert ch==ref[('parent',law,rep)]['coordinate_hash']
                    sparse=np.zeros_like(gt);sparse.reshape(-1)[ids]=gt.reshape(-1)[ids]
                    mask=sparse>0;distance=distance_transform_edt(~mask)/np.sqrt(mask.size/500)
                    regions={'input':valid&mask,'near_unobserved':valid&~mask&(distance<=1),
                             'middle':valid&(distance>1)&(distance<=2),'far':valid&(distance>2)}
                    assert sum(int(m.sum()) for m in regions.values())==int(valid.sum())
                    regions['all']=valid
                    dep=torch.from_numpy(sparse)[None,None].cuda()
                    view=cc.encode(base,image,dep)
                    pred={'parent':arr(cc.propagate(base,view,view['init']))};internal={}
                    keep=row['index'] in plan['exemplars'].values() and li==3 and rep==0
                    for name,c in correctors.items():
                        delta=cc.predict_delta(c,view);initial=cc.corrected(view,delta)
                        pred[name]=arr(cc.propagate(base,view,initial))
                        internal[name]=arr(initial-view['init'])
                    for name,p in pred.items():
                        score=dc.score(p,gt,valid,ids)
                        for metric in dc.METRICS:
                            diff=abs(score[metric]-float(ref[(name,law,rep)][metric]))
                            local_replay=max(local_replay,diff)
                            assert diff<1e-5,(row['index'],law,rep,name,metric,diff)
                    for seed in [42,43,44]:
                        s=pred[f'spatial_seed{seed}'].astype(np.float64)
                        mean=pred[f'image_mean_seed{seed}'].astype(np.float64)
                        parent=pred['parent'].astype(np.float64);truth=gt.astype(np.float64)
                        for control,b in [('image_mean',mean),('parent',parent)]:
                            e=b-truth;c=s-b;es=s-truth
                            for region,m in regions.items():
                                n=int(m.sum());base_sse=float(np.square(e[m]).sum());spatial_sse=float(np.square(es[m]).sum())
                                cross=float((2*e[m]*c[m]).sum());energy=float(np.square(c[m]).sum())
                                error=abs(spatial_sse-base_sse-cross-energy)
                                local_identity=max(local_identity,error)
                                assert np.isclose(spatial_sse-base_sse,cross+energy,rtol=1e-10,atol=1e-7)
                                records.append(dict(index=row['index'],image=row['image'],scene=row['scene'],scan=row['scan'],
                                    law=law,repetition=rep,seed=seed,control=control,region=region,coordinate_hash=ch,
                                    valid_pixels=int(valid.sum()),pixels=n,spatial_sse=spatial_sse,control_sse=base_sse,
                                    delta_sse=spatial_sse-base_sse,cross_term=cross,correction_energy=energy,
                                    opposing_pixels=int((e[m]*c[m]<0).sum()),benefit_pixels=int((es[m]**2<e[m]**2).sum()),
                                    spatial_internal_abs_sum=float(np.abs(internal[f'spatial_seed{seed}'][m]).astype(np.float64).sum()),
                                    mean_internal_abs_sum=float(np.abs(internal[f'image_mean_seed{seed}'][m]).astype(np.float64).sum())))
                    if keep:
                        pixel=pixel_dir/f"{row['index']:04d}_{law}_{rep}.npz"
                        np.savez_compressed(pixel,rgb=rgb,gt=gt,valid=valid,support_indices=ids,distance=distance.astype(np.float32),
                            parent_init=arr(view['init']),spatial_gate=arr(gc.gate(dep,gt.shape,'spatial')),
                            mean_gate=arr(gc.gate(dep,gt.shape,'image_mean')),
                            **{'prediction_'+k:v for k,v in pred.items()},**{'internal_'+k:v for k,v in internal.items()})
                        pixel_hashes[str(pixel)]=dc.sha(pixel)
            assert len(records)==8*5*3*2*5
            tmp=path.with_suffix('.tmp')
            with tmp.open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
            tmp.replace(path)
            maximum_replay=max(maximum_replay,local_replay)
            save(marker,dict(plan_sha256=ph,sha256=dc.sha(path),rows=len(records),replay_max=local_replay,
                identity_max=local_identity,pixel_files=pixel_hashes))
            completed+=1
            status=dict(status='RUNNING',completed=completed,total=len(rows),last_index=row['index'],
                elapsed_seconds=time.monotonic()-start,replay_max=maximum_replay,pid=os.getpid(),plan_sha256=ph)
            save(OUT/'status.json',status);print(json.dumps(status),flush=True)
            if limit and completed>=limit:break
    complete=completed==len(rows)
    save(OUT/'status.json',dict(status='COMPLETE' if complete else 'BOUNDED_REPLAY_COMPLETE',completed=completed,total=len(rows),
        elapsed_seconds=time.monotonic()-start,plan_sha256=ph))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);a=p.parse_args();run(a.limit)
