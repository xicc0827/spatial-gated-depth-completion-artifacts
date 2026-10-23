"""Paper-width 7 inch plots; data from verified reaggregation, no invented intervals."""
from pathlib import Path
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parents[1]/'outputs'
assert any(f.name=='Times New Roman' for f in font_manager.fontManager.ttflist)
plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.labelsize':10,
    'axes.titlesize':11,'legend.fontsize':9,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False})
def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
u=[r for r in readcsv(ROOT/'reanalysis/paired_units.csv') if r['family']=='unseen8' and r['unit']=='scan']
scans=sorted(set(r['id'] for r in u));assert len(scans)==10
fig,axes=plt.subplots(1,3,figsize=(7,4),sharey=True,sharex=True,layout='constrained')
for ax,control,title,color,marker in zip(axes,['parent','image_mean','shifted'],
    ['Parent','Image Mean','Shifted Gate'],['#666666','#0072B2','#D55E00'],['o','s','^']):
    lookup={r['id']:float(r['delta_m']) for r in u if r['control']==control}
    data=np.array([lookup[s] for s in scans])
    ax.axvline(0,color='#999999',lw=.8)
    ax.hlines(np.arange(10),0,data,color=color,lw=1.5)
    ax.scatter(data,np.arange(10),color=color,marker=marker,s=28,zorder=3)
    ax.set_title(f'Spatial vs {title}\n{int((data<0).sum())}/10 scans improved')
    ax.grid(axis='x',alpha=.18);ax.set_yticks(np.arange(10),[s.replace('scan_','') for s in scans])
axes[0].invert_yaxis();axes[0].set_ylabel('Scan')
fig.supxlabel('Difference in mean per-image RMSE (m); negative favors Spatial',fontsize=10)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'unseen8扫描对照.{ext}',dpi=220)
plt.close(fig)
v=next(r for r in readcsv(ROOT/'reanalysis/signed_influence.csv') if r['family']=='unseen8' and r['control']=='image_mean')
values=[float(v[k]) for k in ['group_weighted_contribution_m','remaining_weighted_contribution_m','net_delta_m']]
assert np.isclose(values[0]+values[1],values[2])
fig,ax=plt.subplots(figsize=(7,3.1),layout='constrained')
colors=['#0072B2','#D55E00','#333333']
bars=ax.barh(np.arange(3),values,color=colors,height=.55)
ax.set_yticks(np.arange(3),['Scan 00183 (52 images)','Other scans (252 images)','Net (304 images)']);ax.invert_yaxis()
ax.axvline(0,color='#888888',lw=.8);ax.set_xlim(-.041,.023)
for b,value in zip(bars,values):
    ax.text(value+(-.0008 if value<0 else .0008),b.get_y()+b.get_height()/2,f'{value:+.6f}',
        va='center',ha='right' if value<0 else 'left',fontsize=10)
ax.set_xlabel('Image-weighted contribution to RMSE difference (m)')
ax.set_title('Spatial vs Image Mean | unseen8 | post-hoc decomposition')
ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'unseen8有符号贡献.{ext}',dpi=220)
plt.close(fig)
print('Saved two figures as PNG, PDF and editable SVG.')
