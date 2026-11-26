"""Validation-selected evidence-constrained filtering; no retraining."""
from improve_explainer import *

def main():
    out=Path('outputs/explainer_final'); out.mkdir(parents=True,exist_ok=True)
    bundle=joblib.load('outputs/explainer_verifier/verifier.joblib')
    assert bundle['labels']==LABELS
    def load(p): return pd.read_csv(p,keep_default_na=False)
    def signals(d):
        r=rules(d).astype(bool); x=features(d,bundle['vectorizer'],r.astype(int))
        p=np.column_stack([m.predict_proba(x)[:,1] if m is not None else np.ones(len(d)) for m in bundle['models']])
        return r,p
    def filtered(r,p,t): return r & (p>=t)
    va=load('data/splits/val.csv'); r,p=signals(va); y=truth(va)
    # Only suppress existing evidence, never invent findings. Keep unsupported labels.
    candidates=[]
    for t in [0,.15,.25,.35,.45,.55,.65]:
        m=metrics(y,filtered(r,p,t),va)
        candidates.append((m['f1'],-m['false_alarm'],-t))
    t=-max(candidates)[2]
    print('Validation-selected evidence filter:',t)
    rows=[]; details=[]
    for split,path in [('val','data/splits/val.csv'),('test','data/splits/test.csv'),('behavior','outputs/release_audit/behavior_cases.csv')]:
        d=load(path); r,p=signals(d); y=truth(d)
        for method,pred in [('rules',r),('evidence_filter',filtered(r,p,t))]:
            for lang in ['all','en','hi']:
                mask=np.ones(len(d),bool) if lang=='all' else d.language.eq(lang).to_numpy()
                rows.append(dict(split=split,language=lang,method=method,**metrics(y[mask],pred[mask],d[mask])))
            for j,k in enumerate(LABELS):
                details.append(dict(split=split,method=method,label=k,support=int(y[:,j].sum()),precision=precision_score(y[:,j],pred[:,j],zero_division=0),recall=recall_score(y[:,j],pred[:,j],zero_division=0),f1=f1_score(y[:,j],pred[:,j],zero_division=0)))
    df=pd.DataFrame(rows); df.to_csv(out/'metrics.csv',index=False); pd.DataFrame(details).to_csv(out/'per_label.csv',index=False)
    import json
    (out/'filter_config.json').write_text(json.dumps({'threshold':t,'selection':'validation micro F1 then false alarm then lower threshold','unsupported_labels':'retain rule findings'},indent=2))
    print(df.to_string(index=False)); print('Outputs:',out)
if __name__=='__main__': main()
