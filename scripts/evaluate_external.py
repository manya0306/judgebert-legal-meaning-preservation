from evaluate_release import PATHS,CATS,cat,normalize,encode_frames,equivalence
from improve_explainer import rules,truth,metrics
from pathlib import Path
import argparse,zipfile,json
import numpy as np,pandas as pd,joblib
from sklearn.metrics import accuracy_score,f1_score
from scipy.stats import spearmanr

def main():
 p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--xlmr');p.add_argument('--only-xlmr',action='store_true');args=p.parse_args()
 out=Path('outputs/external_challenge');out.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(args.zip) as z:
  with z.open('data/challenge/challenge_v1.csv') as f:d=pd.read_csv(f,keep_default_na=False)
 assert len(d)==128 and d.pair_id.is_unique and set(d.category)<=set(CATS)
 audits=[];keep=np.ones(len(d),dtype=bool)
 for path in sorted({v[1] for v in PATHS.values()}|{'data/splits/train.csv','data/splits/val.csv','data/splits/test.csv'}):
  t=pd.read_csv(path,keep_default_na=False);hit=np.zeros(len(d),dtype=bool)
  for k in ['pair_id','group_id','concept_id','split_group','original']:
   a,b=d[k].astype(str),t[k].astype(str)
   if k=='original':a,b=a.map(normalize),b.map(normalize)
   h=a.ne('') & a.isin(set(b)-{''});hit|=h.to_numpy();audits.append(dict(dataset=path,field=k,overlap=int(h.sum())))
  keep &= ~hit
 pd.DataFrame(audits).to_csv(out/'overlap.csv',index=False);print(pd.DataFrame(audits).to_string(index=False));d=d.loc[keep].copy();print('Retained:',len(d),'of 128; groups:',d.group_id.nunique(),flush=True);assert len(d)>0
 d.to_csv(out/'challenge_used.csv',index=False)
 rows=[];preds=[]
 def record(name,score,head):
  q=d.copy();q['score']=score;q['head']=head;q['reg_category']=cat(score);q['model']=name;preds.append(q)
  for lang,g in [('all',q),*list(q.groupby('language'))]:
   err=g.score-g.lmp_score;rows.append(dict(model=name,language=lang,n=len(g),mae=abs(err).mean(),rmse=np.sqrt((err**2).mean()),spearman=spearmanr(g.score,g.lmp_score).statistic,reg_accuracy=accuracy_score(g.category,g.reg_category),reg_f1=f1_score(g.category,g.reg_category,labels=CATS,average='macro',zero_division=0),head_accuracy=accuracy_score(g.category,g['head']),head_f1=f1_score(g.category,g['head'],labels=CATS,average='macro',zero_division=0)))
  suffix='xlmr' if name=='xlmr' else 'hybrid';subset=[r for r in rows if (r['model']=='xlmr')==(suffix=='xlmr')];pd.DataFrame(subset).to_csv(out/f'{suffix}_metrics.csv',index=False);pd.concat([r for r in preds if (r.model.iloc[0]=='xlmr')==(suffix=='xlmr')]).to_csv(out/f'{suffix}_predictions.csv',index=False);print(pd.DataFrame(subset).round(4).to_string(index=False),flush=True)
 if not args.only_xlmr:
  r=rules(d);y=truth(d);pd.DataFrame([dict(language=l,**metrics(y[m],r[m],d.loc[m])) for l in ['all','en','hi'] for m in [np.ones(len(d),bool) if l=='all' else d.language.eq(l).to_numpy()]]).to_csv(out/'explainer_metrics.csv',index=False)
  cache=encode_frames([d])
  for name,(path,tr,training,mode) in PATHS.items():
   if not Path(path).exists():print('SKIPPED missing:',path);continue
   obj=joblib.load(path);assert (obj['training'],obj['mode'])==(training,mode);score,head,prob=obj['model'].predict(d,cache);record(name,score,head)
 if args.xlmr:
  import torch
  from lmp.models.train_crossenc import load_checkpoint,predict
  torch.set_num_threads(4);model,tok,meta=load_checkpoint(args.xlmr);score,prob,head=predict(model,tok,d,max_length=meta['max_length'],batch_size=4);record('xlmr',score,head)
 print('Outputs:',out)
if __name__=='__main__':main()
