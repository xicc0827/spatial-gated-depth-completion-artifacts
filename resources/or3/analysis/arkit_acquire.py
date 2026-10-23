"""Official public ARKitScenes metadata and deterministic cohort acquisition."""
from pathlib import Path
import csv,io,json,hashlib,urllib.request,time,zipfile,struct,zlib,concurrent.futures,argparse

ROOT=Path(__file__).resolve().parent/'new_scene_validation'
ROOT.mkdir(exist_ok=True)
PUBLIC={
 'splits.csv':'https://raw.githubusercontent.com/apple-aiml-research/ARKitScenes/main/depth_upsampling/upsampling_train_val_splits.csv',
 'metadata.csv':'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/upsampling/metadata.csv',
 'README.md':'https://raw.githubusercontent.com/apple-aiml-research/ARKitScenes/main/depth_upsampling/README.md',
 'DATA.md':'https://raw.githubusercontent.com/apple-aiml-research/ARKitScenes/main/DATA.md',
 'LICENSE':'https://raw.githubusercontent.com/apple-aiml-research/ARKitScenes/main/LICENSE',
 'dataset_paper.html':'https://arxiv.org/html/2111.08897v2',
}
def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,x):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');q.replace(p)
def request(url,method='GET',headers=None):
 return urllib.request.urlopen(urllib.request.Request(url,method=method,headers={'User-Agent':'ResearchDatasetValidation/1.0',**(headers or {})}),timeout=40)
def metadata():
 folder=ROOT/'source';folder.mkdir(exist_ok=True)
 receipts=[]
 for name,url in PUBLIC.items():
  p=folder/name
  try:
   if not p.exists():
    with request(url) as r:data=r.read()
    p.write_bytes(data)
   receipts.append(dict(name=name,url=url,bytes=p.stat().st_size,sha256=sha(p.read_bytes())))
  except Exception as e:receipts.append(dict(name=name,url=url,error=repr(e)))
 save(folder/'receipts.json',receipts)
 for name in ['metadata.csv','splits.csv']:
  p=folder/name
  if p.exists():
   rows=list(csv.DictReader(p.read_text(encoding='utf-8-sig').splitlines()))
   print(name,'rows',len(rows),'fields',list(rows[0]),'examples',rows[:2],flush=True)
 print(json.dumps(receipts,ensure_ascii=False),flush=True)

def select():
 rows=list(csv.DictReader((ROOT/'source/metadata.csv').read_text(encoding='utf-8-sig').splitlines()))
 eligible={}
 for r in rows:
  if r['fold']=='Validation' and r['sky_direction'] in ('Up','Down'):
   eligible.setdefault(r['visit_id'],[]).append(r)
 visits=sorted(eligible,key=lambda x:sha(('MVA_new_scene_20261021|'+x).encode()))
 chosen=[]
 for i,visit in enumerate(visits):
  r=sorted(eligible[visit],key=lambda r:r['video_id'])[0]
  chosen.append(dict(**r,role='preprocessing_check' if i==0 else ('evaluation' if i<=20 else 'reserve'),rank=i))
 config=dict(status='METADATA_COHORT_SELECTION_BEFORE_PIXELS_OR_PREDICTIONS',created_unix=time.time(),
    official_validation_videos=sum(r['fold']=='Validation' for r in rows),eligible_visits=len(visits),
    ordering='sha256(MVA_new_scene_20261021|visit_id) ascending; lexicographic first upright-compatible video per visit',
    eligibility='Official Validation; sky_direction Up or Down; same visit never crosses preprocessing/evaluation',
    target_visits=20,frames_per_video=15,selected=chosen)
 p=ROOT/'selection.json'
 if p.exists():
  old=json.loads(p.read_text(encoding='utf-8'))
  assert old['selected']==chosen
 else:save(p,config)
 print('eligible visits',len(visits),'selected',chosen[:3])
 return chosen

def probe():
 chosen=select();result=[]
 def one(r):
  url=f"https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/upsampling/Validation/{r['video_id']}.zip"
  with request(url,method='HEAD') as response:
   info=dict(r,url=url,status=response.status,headers=dict(response.headers))
  print(json.dumps({k:info[k] for k in ['visit_id','video_id','role']}|{'bytes':info['headers'].get('Content-Length'),'range':info['headers'].get('Accept-Ranges') }),flush=True)
  return info
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for x in pool.map(one,chosen[:21]):result.append(x)
 save(ROOT/'archive_inventory.json',result)
 print('total bytes',sum(int(x['headers'].get('Content-Length',0)) for x in result))

def download(role):
 folder=ROOT/'archives';folder.mkdir(exist_ok=True)
 rows=json.loads((ROOT/'archive_inventory.json').read_text(encoding='utf-8'))
 rows=[r for r in rows if role=='all' or r['role']==role]
 def one(r):
  p=folder/(r['video_id']+'.zip');receipt=p.with_suffix('.json')
  expected=int(r['headers']['Content-Length'])
  if receipt.exists():
   saved=json.loads(receipt.read_text());assert p.stat().st_size==expected and sha(p.read_bytes())==saved['sha256'];return saved
  start=time.monotonic();tmp=p.with_suffix('.partial')
  for attempt in range(3):
   try:
    with request(r['url']) as response,tmp.open('wb') as f:
     while True:
      chunk=response.read(1024*1024)
      if not chunk:break
      f.write(chunk)
    assert tmp.stat().st_size==expected
    with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
    tmp.replace(p)
    result={k:r[k] for k in ['video_id','visit_id','role','url']}
    result.update(bytes=expected,sha256=sha(p.read_bytes()),seconds=time.monotonic()-start)
    save(receipt,result);print(json.dumps(result),flush=True);return result
   except Exception:
    if attempt==2:raise
    time.sleep(2)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(one,rows))
 save(ROOT/('download_'+role+'.json'),results)
 if role=='preprocessing_check':
  with zipfile.ZipFile(folder/(rows[0]['video_id']+'.zip')) as z:
   print('FIRST NAMES',z.namelist()[:16], 'FILES',len(z.namelist()),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',nargs='?',default='metadata',choices=['metadata','probe','dev','download']);a=p.parse_args()
 if a.action=='metadata':metadata()
 elif a.action=='probe':probe()
 else:download('preprocessing_check' if a.action=='dev' else 'all')
