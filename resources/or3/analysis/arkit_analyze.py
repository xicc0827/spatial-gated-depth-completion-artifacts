"""Full frozen new-venue score validation and pre-specified aggregation."""
from pathlib import Path
import json,hashlib,csv
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent/'new_scene_validation';DEST=ROOT/'summary';DEST.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
plan=read(ROOT/'evaluation_plan_frozen.json');ph=sha(ROOT/'evaluation_plan_frozen.json')
receipt=read(ROOT/'inference_complete.json');assert receipt['plan_sha256']==ph
manifest=read(ROOT/'evaluation_manifest.json')['rows'];manifest=[r for r in manifest if r['primary']]
assert len(manifest)==plan['population_images'] and len({r['visit_id'] for r in manifest})==20
dev=read(ROOT/'development_manifest.json')['rows'];assert not {r['visit_id'] for r in dev}&{r['visit_id'] for r in manifest}
assert sha(ROOT/'evaluation_manifest.json')==plan['manifest_sha256']
assert sha(ROOT/'evaluation_supports.csv')==plan['support_sha256']
support=pd.read_csv(ROOT/'evaluation_supports.csv').set_index(['index','law','repetition'])
dfs=[];regions=[];identity_max=0.;pixel_counts=0
for meta in manifest:
 p=ROOT/'per_image'/f"{meta['index']:04d}.csv";rp=p.with_suffix('.regions.csv');marker=read(p.with_suffix('.json'))
 assert marker['plan_sha256']==ph and marker['sha256']==sha(p)==receipt['files'][str(p)]
 assert marker['region_sha256']==sha(rp)==receipt['files'][str(rp)]
 scores=pd.read_csv(p);reg=pd.read_csv(rp)
 assert len(scores)==385 and len(reg)==1200
 assert set(scores.model)==set(plan['models']) and not scores.duplicated(['model','law','repetition']).any()
 assert set(scores.law)==set(plan['laws']) and set(scores.repetition)==set(range(5))
 assert (scores.input_points==500).all() and (scores.score_pixels==meta['score_pixels']).all()
 assert (scores.scene==meta['scene']).all() and (scores.scan==meta['scan']).all() and (scores.image==meta['image']).all()
 assert np.isfinite(scores[plan['outcomes']].to_numpy()).all() and (scores[plan['outcomes']]>=0).all().all()
 assert scores.groupby(['law','repetition']).coordinate_hash.nunique().eq(1).all()
 for r in scores.drop_duplicates(['law','repetition']).itertuples():assert r.coordinate_hash==support.loc[(meta['index'],r.law,r.repetition),'coordinate_hash']
 assert not reg.duplicated(['law','repetition','seed','control','region']).any()
 assert set(reg.seed)=={42,43,44} and set(reg.control)=={'image_mean','parent'} and set(reg.region)=={'all','far','input','near_unobserved','middle'}
 whole=reg[reg.region=='all']
 old=scores.set_index(['model','law','repetition'])
 for r in whole.itertuples():
  a=old.loc[(f'spatial_seed{r.seed}',r.law,r.repetition),'rmse']**2
  name='parent' if r.control=='parent' else f'image_mean_seed{r.seed}'
  b=old.loc[(name,r.law,r.repetition),'rmse']**2
  assert abs(a-r.spatial_sse/r.valid_pixels)<1e-10 and abs(b-r.control_sse/r.valid_pixels)<1e-10
  pixel_counts+=1
 pivot=reg.pivot(index=['law','repetition','seed','control'],columns='region',values='delta_sse')
 assert np.allclose(pivot['all'],pivot[['input','near_unobserved','middle','far']].sum(axis=1),rtol=1e-10,atol=1e-7)
 assert np.allclose(reg.delta_sse,reg.cross_term+reg.correction_energy,rtol=1e-10,atol=1e-7)
 identity_max=max(identity_max,marker['identity_max']);dfs.append(scores);regions.append(reg)
d=pd.concat(dfs,ignore_index=True);r=pd.concat(regions,ignore_index=True)
assert len(d)==plan['expected_score_rows']==receipt['score_rows']
assert len(r)==len(manifest)*1200==receipt['region_rows']
families={'unseen8':plan['primary_family'],'uniform':['uniform'],'seen2':['left','right']}
long=[];venue_rows=[];perimage_rows=[];paired=[];seed_rows=[];venue_seed=[]
for family,laws in families.items():
 x=d[d.law.isin(laws)].groupby(['index','scene','scan','image','model'],as_index=False)[plan['outcomes']].mean()
 x['method']=x.model.str.replace(r'_seed\d+$','',regex=True)
 x['seed']=x.model.str.extract(r'_seed(\d+)$')[0].fillna('parent')
 x['family']=family;perimage_rows.append(x)
 mean=x.groupby(['index','scene','scan','image','method'],as_index=False)[plan['outcomes']].mean()
 visits=mean.groupby(['scene','method'],as_index=False)[plan['outcomes']].mean();visits['family']=family;venue_rows.append(visits)
 for aggregation in ['image','venue']:
  a=mean if aggregation=='image' else visits
  for method,g in a.groupby('method'):
   long.append(dict(family=family,aggregation=aggregation,method=method,**g[plan['outcomes']].mean().to_dict()))
  wide=a.pivot(index='index' if aggregation=='image' else 'scene',columns='method',values='rmse')
  for control in ['parent','image_mean']:
   diff=wide.spatial-wide[control]
   paired.append(dict(family=family,aggregation=aggregation,control=control,spatial_rmse=wide.spatial.mean(),
    control_rmse=wide[control].mean(),delta_m=diff.mean(),change_percent=100*(wide.spatial.mean()/wide[control].mean()-1),
    wins=int((diff<0).sum()),losses=int((diff>0).sum()),ties=int((diff==0).sum()),units=len(diff)))
 for seed in ['42','43','44']:
  a=x[(x.seed==seed)|(x.method=='parent')].groupby(['scene','method'],as_index=False)[plan['outcomes']].mean()
  a['seed']=int(seed);a['family']=family;venue_seed.append(a)
  wide=a.pivot(index='scene',columns='method',values='rmse')
  for control in ['parent','image_mean']:
   diff=wide.spatial-wide[control]
   seed_rows.append(dict(family=family,seed=int(seed),control=control,spatial_rmse=wide.spatial.mean(),control_rmse=wide[control].mean(),
    delta_m=diff.mean(),change_percent=100*(wide.spatial.mean()/wide[control].mean()-1),venue_wins=int((diff<0).sum())))
pd.DataFrame(long).to_csv(DEST/'aggregate_metrics.csv',index=False,encoding='utf-8-sig')
visits=pd.concat(venue_rows,ignore_index=True);visits.to_csv(DEST/'venue_metrics.csv',index=False,encoding='utf-8-sig')
pd.concat(perimage_rows,ignore_index=True).to_csv(DEST/'per_image_metrics.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(paired).to_csv(DEST/'paired_summary.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(seed_rows).to_csv(DEST/'paired_seeds.csv',index=False,encoding='utf-8-sig')
pd.concat(venue_seed,ignore_index=True).to_csv(DEST/'venue_seed_metrics.csv',index=False,encoding='utf-8-sig')

for name in ['spatial_sse','control_sse','delta_sse','cross_term','correction_energy']:
 r[name.replace('sse','mse') if 'sse' in name else name+'_mse']=r[name]/r.valid_pixels
r['area_fraction']=r.pixels/r.valid_pixels
metrics=['spatial_mse','control_mse','delta_mse','cross_term_mse','correction_energy_mse','area_fraction']
per=r.groupby(['index','scene','seed','control','region'],as_index=False)[metrics].mean()
per.to_csv(DEST/'per_image_seed_regions.csv',index=False,encoding='utf-8-sig')
vs=per.groupby(['scene','seed','control','region'],as_index=False)[metrics].mean()
vs.to_csv(DEST/'venue_seed_regions.csv',index=False,encoding='utf-8-sig')
v=vs.groupby(['scene','control','region'],as_index=False)[metrics].mean()
den=v[v.region=='all'][['scene','control','control_mse']].rename(columns={'control_mse':'whole_control_mse'})
v=v.merge(den,on=['scene','control'],validate='many_to_one')
for col,new in [('delta_mse','contribution_percent'),('cross_term_mse','cross_percent'),('correction_energy_mse','energy_percent')]:v[new]=100*v[col]/v.whole_control_mse
v.to_csv(DEST/'venue_regions.csv',index=False,encoding='utf-8-sig')
overall=v.groupby(['control','region'],as_index=False)[metrics].mean()
den=overall[overall.region=='all'][['control','control_mse']].rename(columns={'control_mse':'whole_control_mse'})
overall=overall.merge(den,on='control',validate='many_to_one')
for col,new in [('delta_mse','contribution_percent'),('cross_term_mse','cross_percent'),('correction_energy_mse','energy_percent')]:overall[new]=100*overall[col]/overall.whole_control_mse
overall.to_csv(DEST/'aggregate_regions.csv',index=False,encoding='utf-8-sig')
cohort=pd.DataFrame([{k:x[k] for k in ['index','scene','scan','score_pixels','score_fraction','mean_depth_m']} for x in manifest])
cohort.groupby('scene').agg(images=('index','size'),score_fraction=('score_fraction','mean'),mean_depth_m=('mean_depth_m','mean')).to_csv(DEST/'cohort_venues.csv',encoding='utf-8-sig')
direct=d[d.law.isin(plan['primary_family'])].groupby(['scene','model']).rmse.mean().groupby('model').mean()
primary_values=pd.DataFrame(long).query("family=='unseen8' and aggregation=='venue'").set_index('method').rmse
direct_values={'parent':direct['parent'],'spatial':np.mean([direct[f'spatial_seed{s}'] for s in [42,43,44]]),
 'image_mean':np.mean([direct[f'image_mean_seed{s}'] for s in [42,43,44]])}
direct_max=max(abs(primary_values[k]-x) for k,x in direct_values.items());assert direct_max<1e-12
validation=dict(images=len(manifest),venues=20,score_rows=len(d),region_rows=len(r),
 plan_sha256=ph,receipt_sha256=sha(ROOT/'inference_complete.json'),all_file_hashes_verified=True,
 exact_condition_grids=True,coordinate_pairing_verified=True,whole_image_mse_checks=pixel_counts,
 partition_and_algebra_verified=True,max_algebra_residual=identity_max,independent_primary_aggregation_max_difference=direct_max,
 time_order_verified=read(ROOT/'evaluation_started.json')['unix']>plan['created_unix'],
 scope='New project-local venue evaluation, prospectively fixed after DIODE analysis; full source-trained methods and fixed inference unchanged')
assert validation['time_order_verified']
(DEST/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
print(json.dumps(validation))
print(pd.DataFrame(paired).to_string(index=False))
print('PAIRED SEEDS');print(pd.DataFrame(seed_rows).query("family=='unseen8'").to_string(index=False))
print('REGIONS');print(overall.query("control=='image_mean'").to_string(index=False))
