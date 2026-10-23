"""Independent checks against original scalar metrics and saved pixel exemplars."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent;M=ROOT/'mechanism'
SRC=Path(r'${DIODE_FREEZE_ROOT}')
plan=json.loads((M/'plan.json').read_text(encoding='utf-8'))
receipt=json.loads((SRC/'inference_complete.json').read_text(encoding='utf-8-sig'))
max_mse=0.;checked=0;pixel_max=0.;pixel_cases=0
for index in plan['indices']:
    p=SRC/'per_frame'/f'{index:04d}.csv'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==receipt['files'][str(p)]
    old=pd.read_csv(p).set_index(['model','law','repetition'])
    new=pd.read_csv(M/'per_image'/p.name);allrows=new[new.region=='all']
    for r in allrows.itertuples():
        a=old.loc[(f'spatial_seed{r.seed}',r.law,r.repetition),'rmse']**2
        name='parent' if r.control=='parent' else f'image_mean_seed{r.seed}'
        b=old.loc[(name,r.law,r.repetition),'rmse']**2
        err=max(abs(a-r.spatial_sse/r.valid_pixels),abs(b-r.control_sse/r.valid_pixels))
        max_mse=max(max_mse,err);assert err<1e-10;checked+=1
    pixel=M/'pixels'/f'{index:04d}_top_0.npz'
    if pixel.exists():
        with np.load(pixel,allow_pickle=False) as a:
            valid=a['valid'];gt=a['gt'].astype(float);dist=a['distance'];obs=np.zeros(valid.shape,dtype=bool);obs.flat[a['support_indices']]=True
            masks={'input':valid&obs,'near_unobserved':valid&~obs&(dist<=1),
                   'middle':valid&(dist>1)&(dist<=2),'far':valid&(dist>2),'all':valid}
            for seed in [42,43,44]:
                es=a[f'prediction_spatial_seed{seed}'].astype(float)-gt
                em=a[f'prediction_image_mean_seed{seed}'].astype(float)-gt
                for region,mask in masks.items():
                    saved=new[(new.seed==seed)&(new.control=='image_mean')&(new.region==region)&(new.law=='top')&(new.repetition==0)].iloc[0]
                    err=abs(float((es[mask]**2-em[mask]**2).sum())-saved.delta_sse)
                    pixel_max=max(pixel_max,err);assert err<1e-7
        pixel_cases+=1
assert checked==72960 and pixel_cases==10
result=dict(original_file_hashes_verified=304,paired_whole_image_mse_checks=checked,
    max_difference_from_original_squared_rmse=max_mse,pixel_exemplars_verified=pixel_cases,
    max_pixel_decomposition_difference=pixel_max)
(M/'summary/independent_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
