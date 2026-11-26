"""Train-only error-label verifier; validation-only operating point selection.
Run from repository root. Does not modify scoring models or existing explanations.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.sparse import hstack, csr_matrix
from lmp.explain.analyzer import analyze
from lmp.models.final_hybrid import evidence

LABELS=['omission','contradiction','changed_obligation','changed_permission','altered_condition','missing_exception','scope_change','altered_dates','altered_numbers','altered_entities','altered_reference']
def rules(d):
    return np.array([[int(k in s) for k in LABELS] for s in ({f.category for f in analyze(r.original,r.simplified,r.language).findings}|{f['category'] for f in evidence(r.original,r.simplified,r.language)} for r in d.itertuples())])
def truth(d):
    return np.array([[int(k in str(s).split('|')) for k in LABELS] for s in d.error_labels])
def features(d,v,r):
    a=v.transform(d.original); b=v.transform(d.simplified)
    delta=a-b; delta.data=np.abs(delta.data)
    return hstack([a,b,delta,csr_matrix(r*3),csr_matrix(np.array([[len(x.split()),len(y.split()),len(y.split())/max(1,len(x.split())),int(l=='hi')] for x,y,l in zip(d.original,d.simplified,d.language)]))],format='csr')
def metrics(y,p,d):
    preserved=d.category.eq('Preserved').to_numpy()
    return dict(precision=precision_score(y,p,average='micro',zero_division=0),recall=recall_score(y,p,average='micro',zero_division=0),f1=f1_score(y,p,average='micro',zero_division=0),false_alarm=float(p[preserved].any(1).mean()) if preserved.any() else np.nan)
def main():
    out=Path('outputs/explainer_verifier'); out.mkdir(parents=True,exist_ok=True)
    load=lambda p:pd.read_csv(p,keep_default_na=False)
    tr=load('data/splits/train.csv'); va=load('data/splits/val.csv')
    v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=18000,sublinear_tf=True).fit(pd.concat([tr.original,tr.simplified]))
    rt,rv=rules(tr),rules(va); xt,xv=features(tr,v,rt),features(va,v,rv); yt,yv=truth(tr),truth(va)
    models=[]
    for j,k in enumerate(LABELS):
        if len(np.unique(yt[:,j]))<2: models.append(None); continue
        models.append(LogisticRegression(C=.5,class_weight='balanced',max_iter=1000,random_state=42).fit(xt,yt[:,j]))
    def probs(x): return np.column_stack([m.predict_proba(x)[:,1] if m is not None else np.zeros(x.shape[0]) for m in models])
    pv=probs(xv); candidates=[]
    for t in [.35,.45,.55,.65,.75,.85]:
        for mode in ['filter','augment']:
            p=(pv>=t)&(rv>0) if mode=='filter' else (pv>=t)
            m=metrics(yv,p,va); candidates.append((m['f1'],-m['false_alarm'],t,mode))
    _,_,t,mode=max(candidates)
    print('Validation-selected threshold and mode:',t,mode)
    joblib.dump(dict(vectorizer=v,models=models,labels=LABELS,threshold=t,mode=mode),out/'verifier.joblib')
    rows=[]; details=[]
    sets={'val':va,'test':load('data/splits/test.csv')}
    bp=Path('outputs/release_audit/behavior_cases.csv')
    if bp.exists(): sets['behavior']=load(bp)
    for split,d in sets.items():
        r=rules(d); y=truth(d); pr=probs(features(d,v,r)); p=(pr>=t)&(r>0) if mode=='filter' else pr>=t
        for method,pred in [('rules',r),('verifier',p)]:
            for lang in ['all','en','hi']:
                mask=np.ones(len(d),bool) if lang=='all' else d.language.eq(lang).to_numpy()
                rows.append(dict(split=split,language=lang,method=method,**metrics(y[mask],pred[mask],d[mask])))
            for j,k in enumerate(LABELS):
                details.append(dict(split=split,method=method,label=k,support=int(y[:,j].sum()),precision=precision_score(y[:,j],pred[:,j],zero_division=0),recall=recall_score(y[:,j],pred[:,j],zero_division=0),f1=f1_score(y[:,j],pred[:,j],zero_division=0)))
    df=pd.DataFrame(rows); df.to_csv(out/'metrics.csv',index=False); pd.DataFrame(details).to_csv(out/'per_label.csv',index=False)
    print(df.to_string(index=False)); print('Outputs:',out)
if __name__=='__main__': main()
