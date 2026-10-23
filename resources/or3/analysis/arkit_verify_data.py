"""Independent native-pixel and support verification, no model predictions."""
from pathlib import Path
import json,zipfile,io,hashlib
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent/'new_scene_validation'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(b):return hashlib.sha256(b).hexdigest()
plan=read(ROOT/'evaluation_plan_frozen.json');manifest=read(ROOT/'evaluation_manifest.json')['rows']
count=0;cases=0;archives=0
for visit in sorted({r['visit_id'] for r in manifest}):
 rows=[r for r in manifest if r['visit_id']==visit];vid=rows[0]['video_id'];p=ROOT/'archives'/(vid+'.zip')
 receipt=read(p.with_suffix('.json'));assert sha(p.read_bytes())==receipt['sha256'];archives+=1
 with zipfile.ZipFile(p) as z:
  files={Path(n).name:n for n in z.namelist() if '/highres_depth/' in n and n.endswith('.png')}
  for r in rows:
   data=z.read(files[r['image']]);assert sha(data)==r['raw_depth_sha256']
   gt=np.asarray(Image.open(io.BytesIO(data)))
   # Exact source locations for nearest-neighbor downscale by6 and original crop.
   yy=(np.arange(228)+6)*6+3;xx=(np.arange(304)+8)*6+3
   if r['sky_direction']=='Down':yy=1439-yy;xx=1919-xx
   source=gt[np.ix_(yy,xx)].astype(np.float32)/1000.
   expected=np.isfinite(source)&(source>.1)&(source<=10);source[~expected]=0
   assert sha(Path(r['converted']).read_bytes())==r['sha256']
   with np.load(r['converted'],allow_pickle=False) as a:
    np.testing.assert_array_equal(a['depth'],source)
    np.testing.assert_array_equal(a['valid'],expected)
    for support in a['support_indices'].reshape(-1,500):
     assert len(np.unique(support))==500 and expected.flat[support].all();cases+=1
   count+=1
result=dict(archives_sha256_checked=archives,native_pixel_maps_verified=count,support_sets_verified=cases,
 depth_units='millimetres to metres',rotation_and_resize_map_verified=True,
 scope='Independent raw source pixel mapping and valid-support verification; no predictions')
(ROOT/'data_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
