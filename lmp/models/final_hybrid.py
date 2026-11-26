"""Frozen multilingual encoder and directional legal features. Research prototype."""
import re
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from lmp.models.sk_models import SKScorer
from lmp.explain.analyzer import extract_features

MODEL='paraphrase-multilingual-MiniLM-L12-v2'
UNITS=r'\b\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|years?|hours?|percent|rupees?)\b|\d+(?:\.\d+)?\s*(?:दिन|सप्ताह|महीने|माह|वर्ष|घंटे|प्रतिशत|रुपये)'
REF=r'(?:section|clause|article|धारा|खंड|अनुच्छेद)\s*\d+[a-zA-Z]?(?:\(\w+\))*'

def evidence(a,b,lang):
    fa,fb=extract_features(a,lang),extract_features(b,lang)
    out=[]
    def add(label,x,y):
        if x!=y: out.append(dict(category=label,original_expression=str(x),simplified_expression=str(y),interpretation='Potential change; verify in context.'))
    for pattern,label in [(UNITS,'altered_numbers'),(REF,'altered_reference')]:
        add(label,sorted(re.findall(pattern,a.lower())),sorted(re.findall(pattern,b.lower())))
    # Only flag order changes when the same recognized parties occur in both texts.
    roles=fa['roles']
    if len(roles)>1 and roles==fb['roles']:
        def order(t): return sorted(roles,key=lambda r:t.lower().find(r.lower()))
        add('altered_entities',order(a),order(b))
    return out

def directional(df):
    rows=[]
    for r in df.itertuples():
        a,b=extract_features(r.original,r.language),extract_features(r.simplified,r.language)
        v=[]
        for k in ['numbers','dates','conditions','exceptions','roles','scope','time_rel','salient']:
            x,y=a[k],b[k]
            if k in ['numbers','dates']: x,y=[z[0] for z in x],[z[0] for z in y]
            x,y=set(x),set(y); v.extend([len(x-y),len(y-x),len(x&y)])
        for k in ['obligation','permission','prohibition','conditional_prohibition','recommendation',None]:
            v.extend([a['modal_class']==k,b['modal_class']==k])
        v.extend([a['negated'],b['negated']])
        e=Counter(x['category'] for x in evidence(r.original,r.simplified,r.language))
        v.extend(e[k] for k in ['altered_numbers','altered_reference','altered_entities'])
        rows.append(v)
    return np.asarray(rows,dtype=float)

class FinalHybrid:
    def __init__(self,mode='hybrid'):
        self.mode=mode
    def raw(self,df,cache):
        a=np.stack([cache[t] for t in df.original]); b=np.stack([cache[t] for t in df.simplified])
        return np.hstack([a,b,abs(a-b),a*b,(a*b).sum(1)[:,None]])
    def features(self,df,cache):
        s=self.pca.transform(self.raw(df,cache))
        if self.mode=='semantic': return s
        l=np.hstack([self.lex._X(df),directional(df)])
        return l if self.mode=='legal_lexical' else np.hstack([s,l])
    def fit(self,df,cache):
        self.lex=SKScorer('lexrule_ridge').fit(df)
        self.pca=PCA(n_components=min(64,len(df)-1),random_state=42).fit(self.raw(df,cache))
        self.scaler=StandardScaler().fit(self.features(df,cache))
        x=self.scaler.transform(self.features(df,cache))
        kw=dict(n_estimators=150,max_depth=3,learning_rate=.05,subsample=.8,random_state=0)
        self.reg=GradientBoostingRegressor(**kw).fit(x,df.lmp_score)
        self.clf=GradientBoostingClassifier(**kw).fit(x,df.category)
        return self
    def predict(self,df,cache):
        x=self.scaler.transform(self.features(df,cache))
        return np.clip(self.reg.predict(x),1,10),self.clf.predict(x),self.clf.predict_proba(x)

def encode_frames(frames):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    enc=SentenceTransformer(MODEL,device='cpu',local_files_only=True)
    texts=sorted({t for df in frames for c in ['original','simplified'] for t in df[c]})
    return dict(zip(texts,enc.encode(texts,batch_size=16,normalize_embeddings=True,show_progress_bar=True)))
