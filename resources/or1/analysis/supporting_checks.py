from pathlib import Path
import json,hashlib,math
import pandas as pd
import numpy as np

OUT=Path(__file__).parent/'reanalysis'
B=Path(r'${DIODE_FREEZE_ROOT}')
C=Path(r'${OMNI_RESULTS_ROOT}')
R=B.parent/'gate_region_diagnostic_20261016'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
    return h.hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

listed={x['path']:x for x in read(C/'FILE_MANIFEST.json')}
assert sha(C/'real_records.csv')==listed['real_records.csv']['sha256']
c=pd.read_csv(C/'real_records.csv',dtype={'scene_id':str,'scan_id':str,'seed':str})
assert len(c)==117040 and c.image_id.nunique()==304
assert not c.duplicated(['image_id','layout','replicate','method','seed']).any()
assert set(zip(c.method,c.seed))=={('Parent','parent'),*{(m,str(s)) for m in ['Spatial','ImageMean'] for s in [42,43,44]}}
assert (c.observed_points==500).all()
metrics=['rmse','mae','rel','unobserved_rmse','unobserved_mae','unobserved_rel']
assert np.isfinite(c[metrics].to_numpy()).all() and (c[metrics]>=0).all().all()
keys=['image_id','layout','replicate']
assert c.groupby(keys).size().eq(7).all() and c.groupby(keys).ngroups==16720
for field in ['sample_sha256','eval_mask_sha256','coordinate_sha256','valid_pixels','scene_id','scan_id']:
    assert c.groupby(keys)[field].nunique().eq(1).all()
mapped={'uniform':'uniform','left':'left','right':'right','top':'top','bottom':'bottom','center_void':'center_hole','diagonal_a':'diag_main_half','diagonal_b':'diag_anti_half','vertical_band':'center_vertical_band','center_disk':'center_disk','opposite_corners':'two_corners'}
bm={r['image']:r for r in read(B/'diode_converted_manifest.json')['rows'] if r['primary']}
sp=pd.read_csv(B/'diode_support_manifest.csv');sm={(int(r['index']),r['law'],int(r['repetition'])):r['coordinate_hash'] for r in sp.to_dict('records')}
for r in c.drop_duplicates(keys).to_dict('records'):
    m=bm[r['image_id']]
    assert r['scene_id']==m['scene'].removeprefix('scene_') and r['scan_id']==m['scan'].removeprefix('scan_')
    assert r['valid_pixels']==m['score_pixels']
    assert r['coordinate_sha256']==sm[(m['index'],mapped[r['layout']],int(r['replicate'])-1)]
families={'unseen8':[x for x in mapped if x not in ['uniform','left','right']],'shifted10':[x for x in mapped if x!='uniform']}
rows=[]
for f,laws in families.items():
    per=c[c.layout.isin(laws)].groupby(['image_id','scene_id','scan_id','method','seed'],as_index=False)[metrics].mean()
    mean=per.groupby(['image_id','scene_id','scan_id','method'],as_index=False)[metrics].mean()
    p=mean.pivot(index=['image_id','scene_id','scan_id'],columns='method',values='rmse').reset_index()
    for control in ['Parent','ImageMean']:
        for agg in ['image','scene','scan']:
            g=p[['Spatial',control]] if agg=='image' else p.groupby(agg+'_id')[['Spatial',control]].mean()
            a,b=g.mean();d=g.Spatial-g[control]
            rows.append({'family':f,'control':control,'aggregation':agg,'spatial_rmse_m':a,'control_rmse_m':b,'delta_m':a-b,'change_percent':100*(a/b-1),'benefit_units':int((d<0).sum()),'harm_units':int((d>0).sum()),'units':len(g)})
pd.DataFrame(rows).to_csv(OUT/'omni_paired_summary.csv',index=False,encoding='utf-8-sig')

logs=[]
for method in ['spatial','image_mean']:
    for seed in [42,43,44]:
        p=C/'training'/f'{method}_{seed}'/'train.jsonl'
        raw=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
        retained=[]
        for x in raw:
            while retained and retained[-1]['step']>=x['step']:retained.pop()
            retained.append(x)
        assert [x['step'] for x in retained]==list(range(1,2001))
        assert all(math.isfinite(x[k]) for x in retained for k in ['loss','depth','geometry','noop','gradient_norm'])
        logs.append({'method':method,'seed':seed,'raw_updates':len(raw),'retained_updates':len(retained),'nonzero_gradient_updates':sum(x['gradient_norm']>0 for x in retained),'gradient_clipped_fraction':np.mean([x['gradient_norm']>1 for x in retained]),'loss_first100':np.mean([x['loss'] for x in retained[:100]]),'loss_last100':np.mean([x['loss'] for x in retained[-100:]]),'path':str(p),'sha256':sha(p)})
pd.DataFrame(logs).to_csv(OUT/'omni_training_checks.csv',index=False,encoding='utf-8-sig')

rplan=read(R/'plan.json');reg=read(B/'model_registry.json')
matched=[]
for item in reg['models']:
    if item['arm'] in ['spatial','image_mean','shifted']:
        assert rplan['input_sha256'][item['checkpoint']]==item['sha256']
        matched.append({'method':item['arm'],'seed':item['seed'],'sha256':item['sha256']})
region_frames=[];region_files=[]
for seed in [42,43,44]:
    folder=R/f'CompletionFormer_seed{seed}';done=read(folder/'complete.json')
    assert sha(R/'plan.json')==done['plan_sha256']
    assert sha(folder/'rows.json')==done['rows_sha256']
    rr=read(folder/'rows.json');assert len(rr)==done['rows']
    region_frames.append(pd.DataFrame(rr))
    region_files.append({'path':str(folder/'rows.json'),'rows':len(rr),'sha256':done['rows_sha256']})
rr=pd.concat(region_frames,ignore_index=True)
assert rr['index'].nunique()==128 and set(rr.law)=={'uniform','left','right'}
rr['sse']=rr.pixels*rr.rmse**2
groupkeys=['seed','index','law','draw','model']
counts=rr.pivot(index=groupkeys,columns='region',values='pixels').fillna(0)
ss=rr.pivot(index=groupkeys,columns='region',values='sse').fillna(0)
assert np.max(np.abs(counts['all']-counts[['near','middle','far']].sum(axis=1)))==0
partition_error=float(np.max(np.abs(ss['all']-ss[['near','middle','far']].sum(axis=1))))
assert np.allclose(ss['all'],ss[['near','middle','far']].sum(axis=1),rtol=1e-12,atol=1e-8)
parts={name:ss[name]/counts['all'] for name in ['input_points','middle','far','all']}
parts['near_unobserved']=(ss['near']-ss['input_points'])/counts['all']
region_rows=[]
for family,laws in {'uniform':['uniform'],'seen2':['left','right']}.items():
    for name,v in parts.items():
        tmp=v.reset_index(name='mse_component');tmp=tmp[tmp.law.isin(laws)]
        means=tmp.groupby('model').mse_component.mean()
        denom=(ss['all']/counts['all']).reset_index(name='mse');denom=denom[(denom.law.isin(laws))&(denom.model=='image_mean')].mse.mean()
        region_rows.append({'cohort':'historically_used_NYUv2_development128','family':family,'component':name,'difference_m2':means['spatial']-means['image_mean'],'contribution_percent_of_image_mean_mse':100*(means['spatial']-means['image_mean'])/denom})
pd.DataFrame(region_rows).to_csv(OUT/'historical_source_region_checks.csv',index=False,encoding='utf-8-sig')
save('supporting_validation.json',{'omni_records':len(c),'omni_source':str(C/'real_records.csv'),'omni_records_sha256':sha(C/'real_records.csv'),'omni_checks':'manifest hash, complete grid, finite metrics, paired hashes/masks, 16720 coordinates matched to B; no rerun of pixel errors','omni_training_logs':logs,'source_region_plan':str(R/'plan.json'),'source_region_scope':rplan['scope'],'region_checkpoint_matches_to_B':matched,'region_files':region_files,'region_sse_partition_max_abs_error':partition_error,'region_scope_limit':'Historical NYUv2 source development. Stored aggregates verified; no reconstruction of masks or per-pixel depth. Cannot replace DIODE mechanism diagnosis or new scenes.'})
seeds=pd.read_csv(OUT/'paired_seeds.csv')
ss=seeds[(seeds.family=='unseen8')&(seeds.control=='image_mean')&(seeds.aggregation=='scene')&(seeds.cohort=='all304')]
print('B paired scene RMSE difference mean, sample SD:',ss.delta_m.mean(),ss.delta_m.std(ddof=1))
loo=pd.read_csv(OUT/'leave_one_unit.csv');print(loo[(loo.family=='unseen8')&(loo.control=='image_mean')&(loo.omitted_unit=='scene')].to_string(index=False))
print('C SAME FAMILY SUMMARY:',pd.DataFrame(rows).query("family=='unseen8'").to_string(index=False))
print('Source region records matched to B:',len(matched),'checkpoints;',len(rr),'region rows')
print(pd.DataFrame(region_rows).to_string(index=False))
