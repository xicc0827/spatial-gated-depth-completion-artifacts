"""New-venue evidence charts, seven-inch width and 10 pt live text."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;S=ROOT/'new_scene_validation/summary';OUT=ROOT.parents[1]/'outputs'
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.titlesize':11,
 'legend.fontsize':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
data=[r for r in rows(S/'venue_metrics.csv') if r['family']=='unseen8']
ids=sorted({r['scene'] for r in data});assert len(ids)==20
lookup={(r['scene'],r['method']):float(r['rmse']) for r in data}
fig,axes=plt.subplots(1,2,figsize=(7,6.1),sharex=True,sharey=True,layout='constrained')
for ax,control,color,marker,label in zip(axes,['parent','image_mean'],['#666666','#0072B2'],['o','s'],['Parent','Image Mean']):
 delta=np.array([lookup[(s,'spatial')]-lookup[(s,control)] for s in ids])
 ax.axvline(0,color='#999999',lw=.8)
 ax.hlines(np.arange(20),0,delta,color=color,lw=1.2);ax.scatter(delta,np.arange(20),s=24,marker=marker,color=color,zorder=3)
 ax.set_title(f'Spatial vs {label}\n{int((delta<0).sum())}/20 venues improved')
 ax.grid(axis='x',alpha=.18);ax.set_yticks(np.arange(20),[s.removeprefix('visit_') for s in ids])
axes[0].invert_yaxis();axes[0].set_ylabel('Venue (visit ID)')
fig.supxlabel('Difference in mean per-image RMSE (m); negative favors Spatial',fontsize=10)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'ARKitScenes逐场所验证.{ext}',dpi=220)
plt.close(fig)

new=[r for r in rows(S/'aggregate_regions.csv') if r['control']=='image_mean']
old=[r for r in rows(ROOT/'mechanism/summary/scene_regions.csv') if r['control']=='image_mean']
regions=['input','near_unobserved','middle','far']
old_den=np.mean([float(r['control_mse']) for r in old if r['region']=='all'])
oldvals={k:100*np.mean([float(r['delta_mse']) for r in old if r['region']==k])/old_den for k in regions}
newvals={r['region']:float(r['contribution_percent']) for r in new}
fig,ax=plt.subplots(figsize=(7,3.8),layout='constrained');neg=np.zeros(2);pos=np.zeros(2)
for key,label,color,hatch in [('input','Input points','#999999',''),('near_unobserved','Near, unobserved','#E69F00','//'),
 ('middle','Middle','#CC79A7','xx'),('far','Far','#0072B2','')]:
 vals=np.array([oldvals[key],newvals[key]])
 ax.bar(np.arange(2),vals,width=.45,bottom=np.where(vals<0,neg,pos),color=color,hatch=hatch,label=label)
 neg+=np.minimum(vals,0);pos+=np.maximum(vals,0)
net=neg+pos;ax.scatter(np.arange(2),net,marker='D',color='black',s=30,label='Net',zorder=4)
ax.set_xticks(np.arange(2),['DIODE\n3 scenes','ARKitScenes\n20 venues'])
ax.axhline(0,color='#999999',lw=.8);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
ax.set_ylabel('Contribution to MSE difference\n(% of Image Mean whole-image MSE)')
ax.set_title('Spatial vs Image Mean | unseen8 | equal scene / venue weights')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,frameon=False)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'新旧场景区域误差对照.{ext}',dpi=220)
plt.close(fig)
print('Saved new-venue RMSE and regional comparison figures.')
