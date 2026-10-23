from pathlib import Path
import json, hashlib, re
import numpy as np
import pandas as pd

SRC=Path(r'${DIODE_FREEZE_ROOT}')
OUT=Path(__file__).parent/'reanalysis'
OUT.mkdir(exist_ok=True)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def save(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
plan=read(SRC/'evaluation_plan_frozen.json')
receipt=read(SRC/'inference_complete.json')
reg=read(SRC/'model_registry.json')
manifest=read(SRC/'diode_converted_manifest.json')
assert sha(SRC/'evaluation_plan_frozen.json')==receipt['plan_sha256']
for name,key in [('model_registry.json','registry_sha256'),('diode_converted_manifest.json','manifest_sha256'),('diode_support_manifest.csv','supports_sha256')]:
    assert sha(SRC/name)==plan[key],name
meta={r['index']:r for r in manifest['rows'] if r['primary']}
expected={'parent':reg['parent']['checkpoint_sha256'],**{f"{r['arm']}_seed{r['seed']}":r['sha256'] for r in reg['models']}}
expected_keys={(m,l,r) for m in expected for l in plan['laws'] for r in range(5)}
metrics=['rmse','mae','rel','unobserved_rmse','unobserved_mae','unobserved_rel']
support=pd.read_csv(SRC/'diode_support_manifest.csv')
supportmap={(int(r['index']),r['law'],int(r['repetition'])):r['coordinate_hash'] for r in support.to_dict('records')}
frames=[]
input_hashes=[]
for name,hash_value in receipt['files'].items():
    p=Path(name)
    assert sha(p)==hash_value,str(p)
    df=pd.read_csv(p)
    assert len(df)==1375
    assert set(zip(df.model,df.law,df.repetition))==expected_keys
    assert not df.duplicated(['index','model','law','repetition']).any()
    assert df['index'].nunique()==1
    idx=int(df['index'].iloc[0]);m=meta[idx]
    for key in ['scene','scan','image']:
        assert df[key].nunique()==1 and df[key].iloc[0]==m[key],(idx,key)
    assert (df.input_points==500).all()
    assert (df.score_pixels==m['score_pixels']).all()
    assert np.isfinite(df[metrics].to_numpy()).all()
    assert (df[metrics]>=0).all().all()
    assert (df.checkpoint_sha256==df.model.map(expected)).all()
    for (law,rep),g in df.groupby(['law','repetition']):
        assert g.coordinate_hash.nunique()==1
        assert g.coordinate_hash.iloc[0]==supportmap[(idx,law,int(rep))]
    frames.append(df)
    input_hashes.append({'path':str(p),'sha256':hash_value,'rows':len(df)})
df=pd.concat(frames,ignore_index=True)
assert len(df)==418000 and set(df['index'])==set(meta)
assert df[['index','law','repetition']].drop_duplicates().shape[0]==16720
print('B records validated: 418000; paired sample sets: 16720',flush=True)
checkpoint_checks=[]
for r in [{'checkpoint':reg['parent']['checkpoint'],'sha256':reg['parent']['checkpoint_sha256'],'arm':'parent','seed':'parent'},*reg['models']]:
    p=Path(r['checkpoint']); row={'method':r['arm'],'seed':r['seed'],'path':str(p),'exists':p.is_file(),'expected_sha256':r['sha256']}
    if p.is_file(): row.update(bytes=p.stat().st_size,sha256=sha(p)); row['hash_match']=row['sha256']==row['expected_sha256']
    checkpoint_checks.append(row)
assert all(r.get('hash_match') for r in checkpoint_checks)
print('B checkpoint identities verified: 25',flush=True)
families={'uniform':['uniform'],'seen2':['left','right'],'unseen8':plan['unseen_layouts'],'shifted10':plan['laws'][1:]}
per_image=[]
for family,laws in families.items():
    g=df[df.law.isin(laws)].groupby(['index','scene','scan','image','model'],as_index=False)[metrics].mean()
    g['family']=family
    per_image.append(g)
per=pd.concat(per_image,ignore_index=True)
per['method']=per.model.str.replace(r'_seed\d+$','',regex=True)
per['seed']=per.model.str.extract(r'_seed(\d+)$')[0].fillna('parent')
per.to_csv(OUT/'per_image_family.csv',index=False,encoding='utf-8-sig')
mean=per.groupby(['family','index','scene','scan','image','method'],as_index=False)[metrics].mean()
aggregates=[]
for (family,method),g in mean.groupby(['family','method']):
    for unit in ['image','scene','scan']:
        v=g[metrics].mean() if unit=='image' else g.groupby(unit)[metrics].mean().mean()
        aggregates.append({'family':family,'method':method,'aggregation':unit,**v.to_dict()})
pd.DataFrame(aggregates).to_csv(OUT/'aggregate_metrics.csv',index=False,encoding='utf-8-sig')
scan=mean.groupby(['family','scene','scan','method'],as_index=False)[metrics].mean()
scene=mean.groupby(['family','scene','method'],as_index=False)[metrics].mean()
scene.to_csv(OUT/'scene_metrics.csv',index=False,encoding='utf-8-sig')
scan.to_csv(OUT/'scan_metrics.csv',index=False,encoding='utf-8-sig')
summary=[]; seedrows=[]; unitrows=[]; loorows=[]; influencerows=[]
for family in families:
    familymean=mean[mean.family==family]
    wide=familymean.pivot(index=['index','scene','scan','image'],columns='method',values='rmse')
    assert len(wide)==304
    for control in ['image_mean','shifted','parent','generic_residual','lora']:
        pair=wide[['spatial',control]].copy().reset_index()
        pair['difference_m']=pair.spatial-pair[control]
        pairs_by_unit={}
        for unit in ['image','scan','scene']:
            values=pair[['spatial',control,'difference_m']] if unit=='image' else pair.groupby(unit)[['spatial',control,'difference_m']].mean()
            pairs_by_unit[unit]=values
            a,b=values.spatial.mean(),values[control].mean()
            summary.append({'family':family,'control':control,'aggregation':unit,'spatial_rmse_m':a,'control_rmse_m':b,'delta_m':a-b,'change_percent':100*(a/b-1),'benefit_units':int((values.difference_m<0).sum()),'harm_units':int((values.difference_m>0).sum()),'tie_units':int((values.difference_m==0).sum()),'units':len(values)})
            if unit in ['scene','scan']:
                for ident,row in values.iterrows():
                    unitrows.append({'family':family,'control':control,'unit':unit,'id':ident,'spatial_rmse_m':row.spatial,'control_rmse_m':row[control],'delta_m':row.difference_m,'change_percent':100*(row.spatial/row[control]-1)})
        # Leave-one analyses reapply each aggregation rule to the remaining units.
        for omitted_unit in ['scene','scan']:
            for omitted in sorted(pair[omitted_unit].unique()):
                remain=pair[pair[omitted_unit]!=omitted]
                for agg in ['image','scene','scan']:
                    vals=remain[['spatial',control]] if agg=='image' else remain.groupby(agg)[['spatial',control]].mean()
                    a,b=vals.spatial.mean(),vals[control].mean()
                    loorows.append({'family':family,'control':control,'omitted_unit':omitted_unit,'omitted_id':omitted,'aggregation':agg,'remaining_images':len(remain),'spatial_rmse_m':a,'control_rmse_m':b,'delta_m':a-b,'change_percent':100*(a/b-1)})
        # Fixed historic diagnostic group, not selected anew to maximize a result.
        h=pair[pair.scan=='scan_00183'];rest=pair[pair.scan!='scan_00183']
        dh,dr=h.difference_m.mean(),rest.difference_m.mean();w=len(h)/len(pair)
        net=pair.difference_m.mean()
        influencerows.append({'family':family,'control':control,'fixed_group':'scan_00183','group_images':len(h),'remaining_images':len(rest),'group_delta_m':dh,'remaining_delta_m':dr,'group_weight':w,'group_weighted_contribution_m':w*dh,'remaining_weighted_contribution_m':(1-w)*dr,'net_delta_m':net,'signed_share_percent':100*w*dh/net if net else None,'zero_crossing_weight':dr/(dr-dh) if dr!=dh else None})
        # Paired seeds; parent is reused as the same single checkpoint, never a new replicate.
        for seed in ['42','43','44']:
            p=per[(per.family==family)&(per.method=='spatial')&(per.seed==seed)]
            q=per[(per.family==family)&(per.method==control)&(per.seed==('parent' if control=='parent' else seed))]
            matched=p[['index','scene','scan','rmse']].merge(q[['index','rmse']],on='index',validate='one_to_one',suffixes=('_spatial','_control'))
            assert len(matched)==304
            for cohort,x in [('all304',matched),('without_scene00019',matched[matched.scene!='scene_00019'])]:
                for agg in ['image','scene','scan']:
                    vals=x[['rmse_spatial','rmse_control']] if agg=='image' else x.groupby(agg)[['rmse_spatial','rmse_control']].mean()
                    a,b=vals.mean()
                    seedrows.append({'family':family,'control':control,'cohort':cohort,'aggregation':agg,'seed':seed,'spatial_rmse_m':a,'control_rmse_m':b,'delta_m':a-b,'change_percent':100*(a/b-1)})
summary=pd.DataFrame(summary);seeds=pd.DataFrame(seedrows)
summary.to_csv(OUT/'paired_summary.csv',index=False,encoding='utf-8-sig')
seeds.to_csv(OUT/'paired_seeds.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(unitrows).to_csv(OUT/'paired_units.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(loorows).to_csv(OUT/'leave_one_unit.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(influencerows).to_csv(OUT/'signed_influence.csv',index=False,encoding='utf-8-sig')
# Independent check: primary scene average directly from condition rows.
direct=df[df.law.isin(families['unseen8'])].groupby(['model','scene']).rmse.mean().groupby('model').mean()
primary_checks=[]
for method in ['parent','spatial','image_mean','shifted','task','task_noop','generic_residual','lora','full_ft']:
    ids=['parent'] if method=='parent' else [f'{method}_seed{s}' for s in (42,43,44)]
    primary_checks.append({'method':method,'mean_rmse_m':float(direct[ids].mean()),'seed_sd_m':float(direct[ids].std(ddof=1)) if method!='parent' else None})
agg=pd.DataFrame(aggregates)
for r in primary_checks:
    v=agg[(agg.family=='unseen8')&(agg.method==r['method'])&(agg.aggregation=='scene')].rmse.item()
    assert abs(v-r['mean_rmse_m'])<1e-12
save('validation.json',{'source':str(SRC),'source_plan_sha256':sha(SRC/'evaluation_plan_frozen.json'),'records':len(df),'images':len(meta),'sample_sets':16720,'scenes':3,'scans':10,'record_files_verified':len(input_hashes),'checkpoint_hashes_verified':len(checkpoint_checks),'checks':'plan/manifest/registry/support hashes; all CSV receipt hashes; full keys; finite nonnegative metrics; metadata; registered checkpoint identity; shared coordinates and pixel counts; direct primary aggregation crosscheck','scope':'Reaggregation of saved metrics; no pixel-level recomputation, inference, or training. Added diagnostics are post hoc.','checkpoint_checks':checkpoint_checks,'input_files':input_hashes,'primary_results':primary_checks})
print(summary[(summary.control=='image_mean')&(summary.family.isin(['unseen8','shifted10']))].to_string(index=False),flush=True)
print(pd.DataFrame(influencerows).query("control=='image_mean' and family in ['unseen8','shifted10']").to_string(index=False),flush=True)
print(seeds.query("control=='image_mean' and family=='unseen8' and aggregation=='scene'").to_string(index=False),flush=True)
