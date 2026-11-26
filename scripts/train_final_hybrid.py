from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score,f1_score
from lmp.models.final_hybrid import FinalHybrid,encode_frames
from lmp.evaluation.metrics import full_metrics
from lmp.preprocess import normalize

def main():
    out=Path('outputs/final_hybrid'); out.mkdir(parents=True,exist_ok=True)
    load=lambda p:pd.read_csv(p,keep_default_na=False)
    train={k:load(p) for k,p in {'original':'data/splits/train.csv','targeted':'data/contrastive_pilot/train_targeted.csv','resampled':'data/contrastive_pilot/train_resampled.csv'}.items()}
    ev={k:load(p) for k,p in {'val':'data/splits/val.csv','test':'data/splits/test.csv','challenge':'data/contrastive_pilot/challenge_new.csv'}.items()}
    for t in train.values():
        for d in ev.values():
            for c in ['pair_id','group_id','split_group']: assert not set(t[c])&set(d[c])
            assert not set(t.original.map(normalize))&set(d.original.map(normalize))
    cache=encode_frames([*train.values(),*ev.values()]); results=[]; preds=[]; best=None
    for k,t in train.items():
        for mode in ['semantic','legal_lexical','hybrid']:
            print(k,mode,flush=True); m=FinalHybrid(mode).fit(t,cache)
            for split,d in ev.items():
                score,cat,prob=m.predict(d,cache); p=d.assign(prediction=score,predicted_category=cat,training=k,model=mode,split=split); preds.append(p)
                for sub,q in [('all',p),('en',p[p.language=='en']),('hi',p[p.language=='hi'])]:
                    metrics=full_metrics(q.lmp_score,q.prediction)
                    row=dict(training=k,model=mode,split=split,subset=sub,**{a:b for a,b in metrics.items() if not isinstance(b,(dict,list))})
                    row['head_accuracy']=accuracy_score(q.category,q.predicted_category); row['head_f1']=f1_score(q.category,q.predicted_category,average='macro',zero_division=0); results.append(row)
                    if split=='val' and sub=='all':
                        key=(row['head_f1'],-row['mae'])
                        if best is None or key>best:
                            best=key; joblib.dump(dict(model=m,training=k,mode=mode),out/'selected.joblib')
            pd.DataFrame(results).to_csv(out/'results.csv',index=False); pd.concat(preds).to_csv(out/'predictions.csv',index=False)
    r=pd.DataFrame(results); print(r[r.subset=='all'][['training','model','split','spearman','mae','f1_macro','head_f1']].to_string(index=False)); print('Selected by validation head F1, then MAE:',best)
if __name__=='__main__': main()
