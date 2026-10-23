from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'mechanism';DEST=SRC/'summary';DEST.mkdir(exist_ok=True)
plan=json.loads((SRC/'plan.json').read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ph=sha(SRC/'plan.json')
parts=[];replay=[];identity=[]
for index in plan['indices']:
    p=SRC/'per_image'/f'{index:04d}.csv';m=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
    assert m['plan_sha256']==ph and m['sha256']==sha(p)
    part=pd.read_csv(p);assert len(part)==1200
    assert not part.duplicated(['law','repetition','seed','control','region']).any()
    assert set(part.law)==set(plan['laws']) and set(part.seed)=={42,43,44}
    for pixel,h in m['pixel_files'].items():assert sha(pixel)==h
    parts.append(part);replay.append(m['replay_max']);identity.append(m['identity_max'])
d=pd.concat(parts,ignore_index=True)
assert len(d)==364800 and d['index'].nunique()==304
keys=['index','law','repetition','seed','control']
v=d.pivot(index=keys,columns='region',values='delta_sse')
assert np.allclose(v['all'],v[['input','near_unobserved','middle','far']].sum(axis=1),rtol=1e-10,atol=1e-7)
assert np.allclose(d.delta_sse,d.cross_term+d.correction_energy,rtol=1e-10,atol=1e-7)
for name in ['spatial_sse','control_sse','delta_sse','cross_term','correction_energy']:
    d[name.replace('sse','mse') if 'sse' in name else name+'_mse']=d[name]/d.valid_pixels
d['area_fraction']=d.pixels/d.valid_pixels
d['opposing_fraction_full']=d.opposing_pixels/d.valid_pixels
d['benefit_fraction_full']=d.benefit_pixels/d.valid_pixels
d['spatial_internal_abs_full']=d.spatial_internal_abs_sum/d.valid_pixels
d['mean_internal_abs_full']=d.mean_internal_abs_sum/d.valid_pixels
metrics=['spatial_mse','control_mse','delta_mse','cross_term_mse','correction_energy_mse',
         'area_fraction','opposing_fraction_full','benefit_fraction_full','spatial_internal_abs_full','mean_internal_abs_full']
per=d.groupby(['index','scene','scan','seed','control','region'],as_index=False)[metrics].mean()
per.to_csv(DEST/'per_image_seed_regions.csv',index=False,encoding='utf-8-sig')
unit=per.groupby(['index','scene','scan','control','region'],as_index=False)[metrics].mean()
outputs={}
for group in ['scene','scan']:
    x=unit.groupby([group,'control','region'],as_index=False)[metrics].mean()
    denominators=x[x.region=='all'][[group,'control','control_mse']].rename(columns={'control_mse':'whole_control_mse'})
    x=x.merge(denominators,on=[group,'control'],validate='many_to_one')
    x['contribution_percent']=100*x.delta_mse/x.whole_control_mse
    x['cross_percent']=100*x.cross_term_mse/x.whole_control_mse
    x['energy_percent']=100*x.correction_energy_mse/x.whole_control_mse
    x['conditional_spatial_rmse']=np.sqrt(x.spatial_mse/x.area_fraction)
    x['conditional_control_rmse']=np.sqrt(x.control_mse/x.area_fraction)
    x['opposing_fraction_region']=x.opposing_fraction_full/x.area_fraction
    x['benefit_fraction_region']=x.benefit_fraction_full/x.area_fraction
    x.to_csv(DEST/f'{group}_regions.csv',index=False,encoding='utf-8-sig');outputs[group]=x
seed=per.groupby(['scene','seed','control','region'],as_index=False)[metrics].mean()
seed.to_csv(DEST/'scene_seed_regions.csv',index=False,encoding='utf-8-sig')
layout=d.groupby(['scene','law','control','region'],as_index=False)[metrics].mean()
layout.to_csv(DEST/'scene_layout_regions.csv',index=False,encoding='utf-8-sig')
checks={'images':304,'rows':len(d),'conditions':304*8*5,'model_outputs':304*8*5*7,
        'replay_max':max(replay),'identity_max':max(identity),'plan_sha256':ph,
        'scope':'post hoc; averaged per-image MSE component, not mean RMSE; each component normalized by all valid pixels before aggregation'}
(DEST/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks))
print(outputs['scene'].query("control=='image_mean'")[['scene','region','contribution_percent','cross_percent','energy_percent','area_fraction']].to_string(index=False))
print('SCAN TOTALS')
print(outputs['scan'].query("control=='image_mean' and region=='all'")[['scan','contribution_percent','cross_percent','energy_percent']].to_string(index=False))
