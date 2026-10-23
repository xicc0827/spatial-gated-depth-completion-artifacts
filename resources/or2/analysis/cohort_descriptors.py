"""Descriptive cohort context; not independent observations or causal tests."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
manifest=json.loads(Path(r'${DIODE_FREEZE_ROOT}\diode_converted_manifest.json').read_text(encoding='utf-8-sig'))
rows=[]
for r in manifest['rows']:
    if not r['primary']:continue
    with np.load(r['converted'],allow_pickle=False) as a:
        gt=a['depth'];valid=a['valid'];z=gt[valid].astype(float)
    rows.append(dict(index=r['index'],scene=r['scene'],scan=r['scan'],valid_pixels=len(z),
        score_fraction=len(z)/gt.size,mean_depth_m=z.mean(),median_depth_m=np.median(z),
        p90_depth_m=np.quantile(z,.9),depth_std_m=z.std(),fraction_over5m=float((z>5).mean())))
d=pd.DataFrame(rows);out=ROOT/'mechanism';d.to_csv(out/'cohort_per_image.csv',index=False,encoding='utf-8-sig')
s=d.groupby('scene').agg(images=('index','size'),score_fraction=('score_fraction','mean'),
    mean_depth_m=('mean_depth_m','mean'),mean_image_p90_m=('p90_depth_m','mean'),
    fraction_over5m=('fraction_over5m','mean'))
s.to_csv(out/'cohort_scene.csv',encoding='utf-8-sig');print(s.to_string())
