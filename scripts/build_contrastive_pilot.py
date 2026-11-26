"""AI-authored synthetic pilot; never modifies existing splits. Not human annotation."""
from pathlib import Path
import sys, hashlib
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lmp.preprocess import tokenize, normalize
from lmp.config import score_to_category
# Each family: domain, English and Hindi (original, faithful near, faithful rewrite,
# consequential near edit, consequential rewrite, partial omission).
# {n} is a deadline in days. Partial scores are provisional severity judgments.
FAMILIES = [
('consumer', [
'The seller must acknowledge a written complaint within {n} days and provide a tracking number.',
'The seller shall acknowledge a written complaint within {n} days and provide a tracking number.',
'Within {n} days of receiving a complaint in writing, the merchant is required to confirm receipt and issue a reference number.',
'The seller may acknowledge a written complaint within {n} days and provide a tracking number.',
'The merchant is free to ignore written grievances and need not issue any reference number.',
'The seller must acknowledge a written complaint within {n} days.'],[
'विक्रेता को लिखित शिकायत की प्राप्ति {n} दिनों के भीतर स्वीकार करनी होगी और एक ट्रैकिंग नंबर देना होगा।',
'विक्रेता को लिखित शिकायत की प्राप्ति {n} दिनों के अंदर स्वीकार करनी होगी और एक ट्रैकिंग नंबर देना होगा।',
'लिखित शिकायत मिलने पर व्यापारी के लिए {n} दिन में उसकी पावती और संदर्भ संख्या देना अनिवार्य है।',
'विक्रेता लिखित शिकायत की प्राप्ति {n} दिनों के भीतर स्वीकार कर सकता है और एक ट्रैकिंग नंबर दे सकता है।',
'व्यापारी लिखित शिकायत को अनदेखा कर सकता है और संदर्भ संख्या देना जरूरी नहीं है।',
'विक्रेता को लिखित शिकायत की प्राप्ति {n} दिनों के भीतर स्वीकार करनी होगी।'], 'changed_obligation'),
('privacy',[
'The processor must delete backup copies within {n} days after consent is withdrawn and confirm deletion in writing.',
'The processor shall delete backup copies within {n} days after consent is withdrawn and confirm deletion in writing.',
'Once permission is revoked, the processor is required to erase backups within {n} days and send written confirmation.',
'The processor may retain backup copies after consent is withdrawn and need not confirm deletion in writing.',
'Withdrawal of permission does not require the processor to erase stored backups or send any confirmation.',
'The processor must delete backup copies within {n} days after consent is withdrawn.'],[
'सहमति वापस लेने के बाद प्रोसेसर को {n} दिनों के भीतर बैकअप प्रतियाँ मिटानी होंगी और लिखित पुष्टि देनी होगी।',
'सहमति वापस लेने के बाद प्रोसेसर को {n} दिनों के अंदर बैकअप प्रतियाँ मिटानी होंगी और लिखित पुष्टि देनी होगी।',
'अनुमति रद्द होने पर प्रोसेसर के लिए {n} दिन में सुरक्षित अतिरिक्त प्रतियाँ हटाना और लिखित प्रमाण देना अनिवार्य है।',
'सहमति वापस लेने के बाद प्रोसेसर बैकअप प्रतियाँ रख सकता है और लिखित पुष्टि देना जरूरी नहीं है।',
'अनुमति रद्द होने पर भी प्रोसेसर को सुरक्षित अतिरिक्त प्रतियाँ हटाने या कोई प्रमाण देने की जरूरत नहीं है।',
'सहमति वापस लेने के बाद प्रोसेसर को {n} दिनों के भीतर बैकअप प्रतियाँ मिटानी होंगी।'], 'changed_obligation'),
('banking',[
'The bank must issue a replacement card within {n} days of a verified request and charge no fee.',
'The bank shall issue a replacement card within {n} days of a verified request and charge no fee.',
'After verifying the application, the bank is required to supply a new card free of charge within {n} days.',
'The bank must issue a replacement card within {m} days of a verified request and charge no fee.',
'The bank may refuse to supply a new card even after verifying the application.',
'The bank must issue a replacement card within {n} days of a verified request.'],[
'सत्यापित अनुरोध के बाद बैंक को {n} दिनों के भीतर नया कार्ड जारी करना होगा और कोई शुल्क नहीं लेना होगा।',
'सत्यापित अनुरोध के बाद बैंक को {n} दिनों के अंदर नया कार्ड जारी करना होगा और कोई शुल्क नहीं लेना होगा।',
'आवेदन की पुष्टि होने पर बैंक के लिए {n} दिन में बिना फीस के प्रतिस्थापन कार्ड देना अनिवार्य है।',
'सत्यापित अनुरोध के बाद बैंक को {m} दिनों के भीतर नया कार्ड जारी करना होगा और कोई शुल्क नहीं लेना होगा।',
'आवेदन की पुष्टि होने पर भी बैंक प्रतिस्थापन कार्ड देने से मना कर सकता है।',
'सत्यापित अनुरोध के बाद बैंक को {n} दिनों के भीतर नया कार्ड जारी करना होगा।'], 'altered_numbers'),
('employment',[
'The employer must reimburse approved travel expenses within {n} days of receiving receipts and send a payment statement.',
'The employer shall reimburse approved travel expenses within {n} days of receiving receipts and send a payment statement.',
'On receipt of the bills, the company is required to repay authorised journey costs within {n} days and issue a record of payment.',
'The employee must reimburse approved travel expenses within {n} days of receiving receipts and send a payment statement.',
'The worker rather than the company is responsible for repaying authorised journey costs and issuing the payment record.',
'The employer must reimburse approved travel expenses within {n} days of receiving receipts.'],[
'रसीदें मिलने के बाद नियोक्ता को {n} दिनों के भीतर स्वीकृत यात्रा खर्च लौटाना होगा और भुगतान विवरण भेजना होगा।',
'रसीदें मिलने के बाद नियोक्ता को {n} दिनों के अंदर स्वीकृत यात्रा खर्च लौटाना होगा और भुगतान विवरण भेजना होगा।',
'बिल प्राप्त होने पर कंपनी के लिए {n} दिन में मंजूर सफर की लागत चुकाना और भुगतान का रिकॉर्ड देना अनिवार्य है।',
'रसीदें मिलने के बाद कर्मचारी को {n} दिनों के भीतर स्वीकृत यात्रा खर्च लौटाना होगा और भुगतान विवरण भेजना होगा।',
'कंपनी के बजाय कामगार मंजूर सफर की लागत चुकाने और भुगतान का रिकॉर्ड देने के लिए जिम्मेदार है।',
'रसीदें मिलने के बाद नियोक्ता को {n} दिनों के भीतर स्वीकृत यात्रा खर्च लौटाना होगा।'], 'altered_entities'),
('environment',[
'The operator must report a chemical spill within {n} days and retain a copy of the report.',
'The operator shall report a chemical spill within {n} days and retain a copy of the report.',
'The facility operator is required to report a chemical leak within {n} days and keep a duplicate of that notification.',
'The operator need not report a chemical spill within {n} days or retain a copy of the report.',
'The facility operator has no duty to report a chemical leak or keep a notification record.',
'The operator must report a chemical spill within {n} days.'],[
'संचालक को रसायन फैलने की सूचना {n} दिनों के भीतर देनी होगी और रिपोर्ट की एक प्रति रखनी होगी।',
'संचालक को रसायन फैलने की सूचना {n} दिनों के अंदर देनी होगी और रिपोर्ट की एक प्रति रखनी होगी।',
'रासायनिक रिसाव होने पर संचालक के लिए {n} दिन में सूचना देना और उसकी नकल सुरक्षित रखना अनिवार्य है।',
'संचालक को रसायन फैलने की सूचना {n} दिनों के भीतर देने या रिपोर्ट की प्रति रखने की जरूरत नहीं है।',
'रासायनिक रिसाव होने पर संचालक सूचना देने या उसका रिकॉर्ड रखने के लिए बाध्य नहीं है।',
'संचालक को रसायन फैलने की सूचना {n} दिनों के भीतर देनी होगी।'], 'changed_obligation'),
('contract',[
'The supplier must replace defective goods within {n} days after inspection and pay return shipping costs.',
'The supplier shall replace defective goods within {n} days after inspection and pay return shipping costs.',
'Following examination, the vendor is required to exchange faulty items within {n} days and cover the cost of sending them back.',
'The supplier must replace defective goods within {m} days after inspection and pay return shipping costs.',
'The vendor may decline to exchange faulty items and leave return transport costs to the buyer.',
'The supplier must replace defective goods within {n} days after inspection.'],[
'निरीक्षण के बाद आपूर्तिकर्ता को {n} दिनों के भीतर दोषपूर्ण सामान बदलना होगा और वापसी का परिवहन खर्च देना होगा।',
'निरीक्षण के बाद आपूर्तिकर्ता को {n} दिनों के अंदर दोषपूर्ण सामान बदलना होगा और वापसी का परिवहन खर्च देना होगा।',
'जाँच के बाद विक्रेता के लिए {n} दिन में खराब वस्तुएँ बदलना और उन्हें वापस भेजने की लागत चुकाना अनिवार्य है।',
'निरीक्षण के बाद आपूर्तिकर्ता को {m} दिनों के भीतर दोषपूर्ण सामान बदलना होगा और वापसी का परिवहन खर्च देना होगा।',
'विक्रेता खराब वस्तुएँ बदलने से मना कर सकता है और वापसी की लागत खरीदार पर डाल सकता है।',
'निरीक्षण के बाद आपूर्तिकर्ता को {n} दिनों के भीतर दोषपूर्ण सामान बदलना होगा।'], 'altered_numbers'),
]
# Last two families are held out entirely; no shared template across partitions.
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'/'contrastive_pilot'; OUT.mkdir(parents=True,exist_ok=True)
base={s:pd.read_csv(ROOT/'data'/'splits'/f'{s}.csv',keep_default_na=False) for s in ('train','val','test')}
cols=base['train'].columns
rows={'train':[], 'challenge':[]}
for i,(domain,en,hi,err) in enumerate(FAMILIES):
    split='train' if i<4 else 'challenge'
    for n in (7,14,28):
        group=f'PILOT-{i:02d}'
        concept=f'{group}-{n}'
        for lang, texts in [('en',en),('hi',hi)]:
            texts=[t.format(n=n,m=n*2) for t in texts]
            for j,(score,label,strategy) in enumerate([(9,'','near_faithful'),(9,'','rewrite_faithful'),(2,err,'near_change'),(2,'changed_obligation|omission','rewrite_change'),(6,'omission','partial_omission')],1):
                row=dict.fromkeys(cols,'')
                row.update(pair_id=f'{concept}-{lang}-{j}',language=lang,legal_domain=domain,original=texts[0],simplified=texts[j],lmp_score=score,category=score_to_category(score),error_labels=label,source_doc_id=group,group_id=group,split_group=group,concept_id=concept,data_origin='ai_generated',dataset_tier='ai_generated',generation_strategy=strategy,generator='ChatGPT AI-authored template pilot',label_source='ai_generation_intent',annotation_status='AI provisional; not independently reviewed',source_provenance='Invented legal-style provision; not actual law',license_note='Project-generated synthetic text')
                rows[split].append(row)
frames={k:pd.DataFrame(v,columns=cols) for k,v in rows.items()}
def keys(df): return set(zip(df.language,df.original.map(normalize),df.simplified.map(normalize)))
def originals(df): return set(zip(df.language,df.original.map(normalize)))
for k,d in frames.items():
    assert len(keys(d))==len(d) and d.pair_id.is_unique
    for b in base.values():
        assert not keys(d)&keys(b) and not originals(d)&originals(b)
assert not originals(frames['train'])&originals(frames['challenge'])
assert not set(frames['train'].group_id)&set(frames['challenge'].group_id)
for k,d in frames.items(): d.to_csv(OUT/f'{k}_new.csv',index=False,encoding='utf-8')
pd.concat([base['train'],frames['train']],ignore_index=True).to_csv(OUT/'train_targeted.csv',index=False,encoding='utf-8')
for k,d in [('original_train',base['train']),('targeted_train',pd.concat([base['train'],frames['train']])),*frames.items()]:
    def overlap(r):
        a,b=set(tokenize(r.original)),set(tokenize(r.simplified)); return len(a&b)/max(1,len(a|b))
    x=d.copy(); x['overlap']=x.apply(overlap,axis=1); x['bin']=pd.cut(x.overlap,[-.01,.4,.7,1],labels=['low','medium','high']); print('\n',k,len(x)); print(pd.crosstab(x.category,x.bin)); print('Spearman:',x[['overlap','lmp_score']].corr(method='spearman').iloc[0,1])
print('Exact pair/original isolation checks PASS; semantic overlap and labels require review.')
