from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score,accuracy_score
from scipy.stats import spearmanr
from lmp.models.final_hybrid import encode_frames,evidence
from lmp.models.legal_equivalence import equivalence
from lmp.explain.analyzer import analyze
from lmp.preprocess import normalize
from lmp.human_validation import ingest

CATS=['Not Preserved','Partially Preserved','Preserved']
# Authored diagnostic cases, not independently human annotated. Each tuple has EN and HI versions.
CASES=[
('faithful_passive',2,'','The registrar must issue a receipt to each applicant.','Each applicant must be issued a receipt by the registrar.','पंजीयक को प्रत्येक आवेदक को रसीद देनी होगी।','प्रत्येक आवेदक को पंजीयक द्वारा रसीद दी जानी अनिवार्य है।'),
('faithful_reorder',2,'','After inspection, the inspector must sign the register.','The inspector must sign the register after inspection.','निरीक्षण के बाद निरीक्षक को रजिस्टर पर हस्ताक्षर करने होंगे।','निरीक्षक को रजिस्टर पर हस्ताक्षर निरीक्षण के बाद करने होंगे।'),
('faithful_modal',2,'','The supplier shall deliver the goods.','The supplier must deliver the goods.','आपूर्तिकर्ता को माल पहुँचाना होगा।','आपूर्तिकर्ता के लिए माल पहुँचाना अनिवार्य है।'),
('identity',2,'','The board may appoint an auditor.','The board may appoint an auditor.','बोर्ड लेखापरीक्षक नियुक्त कर सकता है।','बोर्ड लेखापरीक्षक नियुक्त कर सकता है।'),
('obligation',0,'changed_obligation','Every witness is required to disclose conflicts of interest.','Every witness is permitted to disclose conflicts of interest.','प्रत्येक गवाह को हितों के टकराव का खुलासा करना अनिवार्य है।','प्रत्येक गवाह हितों के टकराव का खुलासा कर सकता है।'),
('permission',0,'changed_permission','The trustee may seek an extension.','The trustee is required to seek an extension.','न्यासी समय विस्तार माँग सकता है।','न्यासी को समय विस्तार माँगना होगा।'),
('negation',0,'contradiction','No visitor may enter the archive.','Visitors may enter the archive.','किसी आगंतुक को अभिलेखागार में प्रवेश की अनुमति नहीं है।','आगंतुकों को अभिलेखागार में प्रवेश की अनुमति है।'),
('condition',0,'altered_condition','A refund is payable only if the permit is surrendered.','A refund is payable only if the permit is renewed.','धनवापसी तभी देय है जब परमिट वापस किया जाए।','धनवापसी तभी देय है जब परमिट का नवीनीकरण किया जाए।'),
('exception',0,'missing_exception','All records must be disclosed except sealed records.','All records must be disclosed.','सीलबंद अभिलेखों को छोड़कर सभी अभिलेखों का खुलासा करना होगा।','सभी अभिलेखों का खुलासा करना होगा।'),
('scope',0,'scope_change','Every branch must maintain an emergency register.','Some branches must maintain an emergency register.','प्रत्येक शाखा को आपातकालीन रजिस्टर रखना होगा।','कुछ शाखाओं को आपातकालीन रजिस्टर रखना होगा।'),
('date',0,'altered_dates','The licence expires on 15 March 2028.','The licence expires on 15 May 2028.','लाइसेंस 15 मार्च 2028 को समाप्त होगा।','लाइसेंस 15 मई 2028 को समाप्त होगा।'),
('number',0,'altered_numbers','The deposit is 750 rupees per application.','The deposit is 570 rupees per application.','प्रति आवेदन जमा राशि 750 रुपये है।','प्रति आवेदन जमा राशि 570 रुपये है।'),
('unit',0,'altered_numbers','The notice must remain displayed for 48 hours.','The notice must remain displayed for 48 days.','सूचना 48 घंटे तक प्रदर्शित रहनी चाहिए।','सूचना 48 दिन तक प्रदर्शित रहनी चाहिए।'),
('entity',0,'altered_entities','The borrower must reimburse the lender.','The lender must reimburse the borrower.','उधारकर्ता को ऋणदाता को प्रतिपूर्ति करनी होगी।','ऋणदाता को उधारकर्ता को प्रतिपूर्ति करनी होगी।'),
('reference',0,'altered_reference','Appeals are governed by section 17(2).','Appeals are governed by section 19(2).','अपीलें धारा 17(2) द्वारा शासित हैं।','अपीलें धारा 19(2) द्वारा शासित हैं।'),
('omission',1,'omission','The custodian must inventory the equipment and notify the board.','The custodian must inventory the equipment.','अभिरक्षक को उपकरणों की सूची बनानी होगी और बोर्ड को सूचित करना होगा।','अभिरक्षक को उपकरणों की सूची बनानी होगी।'),
('partial_detail',1,'omission','The report must include the inspection findings and the inspector’s signature.','The report must include the inspection findings.','रिपोर्ट में निरीक्षण के निष्कर्ष और निरीक्षक के हस्ताक्षर होने चाहिए।','रिपोर्ट में निरीक्षण के निष्कर्ष होने चाहिए।'),
('faithful_synonym',2,'','The committee is authorized to reject incomplete applications.','The committee has permission to refuse incomplete applications.','समिति को अधूरे आवेदन अस्वीकार करने का अधिकार है।','समिति अधूरे आवेदन नामंजूर कर सकती है।'),
]
PATHS={'v1_targeted':('outputs/final_hybrid/selected.joblib','data/contrastive_pilot/train_targeted.csv','targeted','hybrid'),'v2_targeted':('outputs/final_human_private/v2_targeted.joblib','data/v2/train_targeted.csv','targeted','hybrid'),'v2_selected':('outputs/final_hybrid_v2/selected.joblib','data/splits/train.csv','original','legal_lexical')}
def labels(s): return set(str(s).split('|'))-{''}
def cat(s): return np.where(s<4,CATS[0],np.where(s<7.5,CATS[1],CATS[2]))
def main():
    out=Path('outputs/release_audit'); out.mkdir(parents=True,exist_ok=True)
    private=Path('outputs/final_human_private')
    d=pd.DataFrame([dict(pair_id=f'behavior_{i}_{lang}',case=k,language=lang,original=a,simplified=b,category=CATS[c],error_labels=e) for i,(k,c,e,en1,en2,hi1,hi2) in enumerate(CASES) for lang,a,b in [('en',en1,en2),('hi',hi1,hi2)]])
    test=pd.read_csv('data/splits/test.csv',keep_default_na=False)
    d.to_csv(out/'behavior_cases.csv',index=False)
    cache=encode_frames([d,test]); rows=[]; preds=[]; expl=[]
    for split,frame in [('behavior',d),('original_test',test)]:
        for r in frame.itertuples():
            base={f.category for f in analyze(r.original,r.simplified,r.language).findings}
            extra={f['category'] for f in evidence(r.original,r.simplified,r.language)}
            expl.append(dict(split=split,pair_id=r.pair_id,language=r.language,category=r.category,expected=r.error_labels,base='|'.join(sorted(base)),enhanced='|'.join(sorted(base|extra))))
    e=pd.DataFrame(expl); e.to_csv(out/'explanation_predictions.csv',index=False)
    er=[]
    for split,g in e.groupby('split'):
        for lang,q in [('all',g),*list(g.groupby('language'))]:
            for method in ['base','enhanced']:
                pairs=[(labels(a),labels(b)) for a,b in zip(q.expected,q[method])]
                for label in ['ALL',*sorted(set().union(*(a|b for a,b in pairs)))]:
                    p=[(a,b) if label=='ALL' else (a&{label},b&{label}) for a,b in pairs]; tp=sum(len(a&b) for a,b in p); fp=sum(len(b-a) for a,b in p); fn=sum(len(a-b) for a,b in p)
                    er.append(dict(split=split,language=lang,method=method,label=label,tp=tp,fp=fp,fn=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn),preserved_false_alarm_rate=float(q.loc[q.category==CATS[2],method].ne('').mean()) if label=='ALL' else np.nan))
    pd.DataFrame(er).to_csv(out/'explanation_metrics.csv',index=False)
    for name,(path,tr,training,mode) in PATHS.items():
        obj=joblib.load(path); assert (obj['training'],obj['mode'])==(training,mode)
        t=pd.read_csv(tr,keep_default_na=False)
        for split,frame in [('behavior',d),('original_test',test)]:
            score,head,prob=obj['model'].predict(frame,cache); p=frame.copy(); p['score']=score;p['head']=head;p['score_category']=cat(score);p['max_probability']=prob.max(1);p['equivalence']=[equivalence(a,b,l) or '' for a,b,l in zip(p.original,p.simplified,p.language)];p['system_head']=np.where(p.equivalence.ne(''),CATS[2],head);p['system_score']=np.where(p.equivalence.ne(''),10,score);p['overlap']=p.original.map(normalize).isin(set(t.original.map(normalize)));p['model']=name;p['split']=split;preds.append(p)
            for lang,q in [('all',p),*list(p.groupby('language'))]:
                rows.append(dict(model=name,split=split,language=lang,n=len(q),original_overlap=int(q.overlap.sum()),head_accuracy=accuracy_score(q.category,q['head']),head_f1=f1_score(q.category,q['head'],labels=CATS,average='macro',zero_division=0),system_f1=f1_score(q.category,q.system_head,labels=CATS,average='macro',zero_division=0),disagreement=float((q['head']!=q.score_category).mean()),low_probability=float((q.max_probability<.6).mean()),false_equivalence=int((q.equivalence.ne('') & q.category.ne(CATS[2])).sum())))
    pd.concat(preds).to_csv(out/'behavior_and_test_predictions.csv',index=False); r=pd.DataFrame(rows);r.to_csv(out/'behavior_metrics.csv',index=False);print(r.round(4).to_string(index=False))
    # Paired concept-group bootstrap on existing private human predictions. No model selection or tuning.
    L=ingest.load_and_validate(Path('outputs/human_validation_v2/completed'),ingest.DesignSpec(),Path.cwd());assert L.ok
    c=L.consensus; h=pd.read_csv(private/'predictions_private.csv'); rng=np.random.default_rng(42); boot=[]
    for target,col,base in [('student','student_mean',c),('expert','expert_score',c[c.expert_reviewed])]:
        q=base[base.dataset_split=='test'].merge(h.pivot(index='pair_id',columns='model_name',values='score'),on='pair_id',validate='one_to_one'); assert len(q)==(47 if target=='student' else 20)
        groups=[np.asarray(v) for v in q.groupby('group_id').indices.values()]; y=q[col].to_numpy(float)
        for a,b in [('v1_targeted','v2_selected'),('v2_targeted','v2_selected'),('v2_targeted','v1_targeted')]:
            x,z=q[a].to_numpy(),q[b].to_numpy()
            def diff(ix):
                return [np.mean(abs(y[ix]-z[ix]))-np.mean(abs(y[ix]-x[ix])),spearmanr(y[ix],x[ix]).statistic-spearmanr(y[ix],z[ix]).statistic]
            vals=np.asarray([diff(np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])) for _ in range(1000)])
            for j,m in enumerate(['mae_improvement','spearman_improvement']):
                v=vals[:,j];v=v[np.isfinite(v)];lo,hi=np.quantile(v,[.025,.975]) if len(v) else (np.nan,np.nan);boot.append(dict(target=target,candidate=a,comparison=b,metric=m,n=len(q),groups=len(groups),estimate=diff(np.arange(len(q)))[j],ci_low=lo,ci_high=hi,valid_replicates=len(v)))
    pd.DataFrame(boot).to_csv(out/'human_paired_bootstrap.csv',index=False);print(pd.DataFrame(boot).round(4).to_string(index=False));print('Audit outputs:',out)
if __name__=='__main__': main()
