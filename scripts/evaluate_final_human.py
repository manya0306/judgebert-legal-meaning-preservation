from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import f1_score,accuracy_score
from lmp.models.final_hybrid import FinalHybrid,encode_frames
from lmp.human_validation import ingest,model_eval
from lmp.human_validation.stats import subset_masks
from lmp.preprocess import normalize
from lmp.evaluation.metrics import full_metrics

def main():
    root=Path.cwd(); out=root/'outputs/final_human_private'; out.mkdir(parents=True,exist_ok=True)
    L=ingest.load_and_validate(root/'outputs/human_validation_v2/completed',ingest.DesignSpec(),root)
    if not L.ok: raise RuntimeError(L.report.to_markdown())
    cons=L.consensus.copy(); df=cons[['pair_id','original','simplified','language']].copy()
    paths={'v1_targeted':'data/contrastive_pilot/train_targeted.csv','v2_targeted':'data/v2/train_targeted.csv','v2_selected':'data/splits/train.csv'}
    train={k:pd.read_csv(p,keep_default_na=False) for k,p in paths.items()}
    cache=encode_frames([df,*train.values()]); rows=[]; predictions=[]; heads=[]
    for name,t in train.items():
        print(name,flush=True)
        if name=='v1_targeted':
            obj=joblib.load('outputs/final_hybrid/selected.joblib'); assert obj['training']=='targeted' and obj['mode']=='hybrid'; m=obj['model']
        elif name=='v2_selected':
            obj=joblib.load('outputs/final_hybrid_v2/selected.joblib'); assert obj['training']=='original' and obj['mode']=='legal_lexical'; m=obj['model']
        else:
            m=FinalHybrid('hybrid').fit(t,cache); joblib.dump(dict(model=m,training='targeted',mode='hybrid'),out/'v2_targeted.joblib')
        score,cat,prob=m.predict(df,cache)
        p=cons.copy(); p['score']=score; p['head']=cat
        overlap=p.source_pair_id.isin(t.pair_id)|p.original.map(normalize).isin(set(t.original.map(normalize)))
        for c in ['group_id','concept_id']:
            overlap |= p[c].astype(str).isin(set(t[c].astype(str))-{''})
        p['training_overlap']=overlap
        predictions.append(p[['pair_id','score','head','training_overlap']].assign(model_name=name))
        for target,col,base in [('student_mean','student_mean',p),('expert','expert_score',p[p.expert_reviewed])]:
            for sub,mask in subset_masks(base,('all','fixed_test','supplemental_train_val','english','hindi','english_fixed_test','hindi_fixed_test')).items():
                q=base[mask];
                if len(q)<3: continue
                y=q[col].to_numpy(float); metrics=full_metrics(y,q.score.to_numpy(float),ingest.HARMONIZED)
                rows.append(dict(model=name,target=target,subset=sub,n=len(q),training_overlap=int(q.training_overlap.sum()),**{k:metrics[k] for k in ['mae','rmse','spearman','pearson','bias','accuracy','f1_macro']}))
                # Human categories use harmonized cutoffs, unlike training categories.
                truth=np.where(y<3.5,'Not Preserved',np.where(y<6.5,'Partially Preserved','Preserved'))
                heads.append(dict(model=name,target=target,subset=sub,n=len(q),training_overlap=int(q.training_overlap.sum()),head_accuracy=accuracy_score(truth,q['head']),head_f1=f1_score(truth,q['head'],labels=['Not Preserved','Partially Preserved','Preserved'],average='macro',zero_division=0)))
    pd.concat(predictions).to_csv(out/'predictions_private.csv',index=False)
    r=pd.DataFrame(rows); h=pd.DataFrame(heads); r.to_csv(out/'regression_metrics.csv',index=False); h.to_csv(out/'head_metrics.csv',index=False)
    print(r[r.subset.isin(['all','fixed_test'])].round(4).to_string(index=False)); print(h[h.subset.isin(['all','fixed_test'])].round(4).to_string(index=False))
    print('Private outputs:',out)
if __name__=='__main__': main()
