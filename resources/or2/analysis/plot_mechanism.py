"""Full-cohort diagnostic plot; signed contributions preserve additive MSE units."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parents[1]/'outputs'
with (ROOT/'mechanism/summary/scene_regions.csv').open(encoding='utf-8-sig',newline='') as f:
    data=[r for r in csv.DictReader(f) if r['control']=='image_mean']
scenes=['scene_00019','scene_00020','scene_00021']
lookup={(r['scene'],r['region']):r for r in data}
plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.titlesize':11,
    'legend.fontsize':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(7,4.2),sharey=True,layout='constrained')
terms=[('input','Input points','#999999',''),('near_unobserved','Near, unobserved','#E69F00','//'),
       ('middle','Middle','#CC79A7','xx'),('far','Far','#0072B2','')]
neg=np.zeros(3);pos=np.zeros(3)
for key,label,color,hatch in terms:
    vals=np.array([float(lookup[(s,key)]['contribution_percent']) for s in scenes])
    axes[0].bar(np.arange(3),vals,bottom=np.where(vals<0,neg,pos),width=.55,color=color,hatch=hatch,label=label)
    neg+=np.minimum(vals,0);pos+=np.maximum(vals,0)
totals=np.array([float(lookup[(s,'all')]['contribution_percent']) for s in scenes])
axes[0].scatter(np.arange(3),totals,marker='D',s=25,color='black',label='Net',zorder=4)
neg=np.zeros(3);pos=np.zeros(3)
for key,label,color,hatch in [('cross_percent','Error–correction cross term','#009E73',''),('energy_percent','Squared correction term','#D55E00','//')]:
    vals=np.array([float(lookup[(s,'all')][key]) for s in scenes])
    axes[1].bar(np.arange(3),vals,bottom=np.where(vals<0,neg,pos),width=.55,color=color,hatch=hatch,label=label)
    neg+=np.minimum(vals,0);pos+=np.maximum(vals,0)
axes[1].scatter(np.arange(3),totals,marker='D',s=25,color='black',label='Net',zorder=4)
for ax in axes:
    ax.axhline(0,color='#777777',lw=.8);ax.set_xticks(np.arange(3),['00019','00020','00021'])
    ax.set_xlabel('Physical scene');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),frameon=False,ncol=1)
axes[0].set_ylabel('Contribution to MSE difference\n(% of Image Mean whole-image MSE)')
axes[0].set_title('(a) Distance regions');axes[1].set_title('(b) Output-difference decomposition')
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'DIODE误差区域与修正分解.{ext}',dpi=220)
print('Saved full-cohort mechanism figure.')
