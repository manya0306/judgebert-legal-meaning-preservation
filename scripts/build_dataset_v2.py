from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from lmp.preprocess import normalize
# Each row: domain, actor EN/HI, action EN/HI, object EN/HI, alternative actor EN/HI.
# Synthetic provisions, not quotations from legislation.
SPECS='''housing|housing board|आवास बोर्ड|publish|प्रकाशित|allocation list|आवंटन सूची|applicant|आवेदक
insurance|insurer|बीमाकर्ता|provide|प्रदान|claim receipt|दावा रसीद|policyholder|बीमाधारक
education|registrar|कुलसचिव|issue|जारी|enrolment certificate|नामांकन प्रमाणपत्र|student|विद्यार्थी
transport|carrier|वाहक|submit|जमा|inspection record|निरीक्षण रिकॉर्ड|passenger|यात्री
healthcare|clinic|क्लिनिक|provide|प्रदान|discharge summary|छुट्टी का सारांश|patient|मरीज
tax|taxpayer|करदाता|submit|जमा|income declaration|आय घोषणा|auditor|लेखापरीक्षक
procurement|committee|समिति|publish|प्रकाशित|award notice|ठेका सूचना|bidder|बोलीदाता
energy|licensee|लाइसेंसधारी|submit|जमा|meter report|मीटर रिपोर्ट|consumer|उपभोक्ता
telecom|provider|सेवा प्रदाता|issue|जारी|usage statement|उपयोग विवरण|subscriber|ग्राहक
copyright|publisher|प्रकाशक|provide|प्रदान|royalty statement|रॉयल्टी विवरण|author|लेखक
municipal|council|परिषद|publish|प्रकाशित|budget summary|बजट सारांश|resident|निवासी
food|manufacturer|निर्माता|submit|जमा|safety certificate|सुरक्षा प्रमाणपत्र|retailer|खुदरा विक्रेता
shipping|shipowner|जहाज मालिक|submit|जमा|cargo manifest|माल सूची|consignee|प्राप्तकर्ता
pension|trustee|न्यासी|provide|प्रदान|benefit statement|लाभ विवरण|beneficiary|लाभार्थी
construction|contractor|ठेकेदार|submit|जमा|completion report|पूर्णता रिपोर्ट|architect|वास्तुकार
research|sponsor|प्रायोजक|publish|प्रकाशित|trial summary|परीक्षण सारांश|investigator|अन्वेषक
customs|importer|आयातक|submit|जमा|origin certificate|मूल प्रमाणपत्र|broker|दलाल
cooperative|secretary|सचिव|issue|जारी|membership receipt|सदस्यता रसीद|member|सदस्य
water|utility|जल सेवा संस्था|provide|प्रदान|quality report|गुणवत्ता रिपोर्ट|householder|गृहस्वामी
licensing|authority|प्राधिकरण|issue|जारी|renewal notice|नवीनीकरण सूचना|licensee|लाइसेंसधारी
charity|foundation|प्रतिष्ठान|publish|प्रकाशित|donation report|दान रिपोर्ट|donor|दाता
agriculture|warehouse|गोदाम संस्था|issue|जारी|storage receipt|भंडारण रसीद|farmer|किसान
aviation|airport operator|हवाई अड्डा संचालक|submit|जमा|runway inspection report|रनवे निरीक्षण रिपोर्ट|airline|विमान सेवा
securities|issuer|निर्गमकर्ता|publish|प्रकाशित|dividend notice|लाभांश सूचना|investor|निवेशक
maritime|port authority|बंदरगाह प्राधिकरण|issue|जारी|berthing permit|बर्थ अनुमति पत्र|ship agent|जहाज एजेंट
heritage|museum|संग्रहालय|publish|प्रकाशित|acquisition register|अधिग्रहण रजिस्टर|visitor|आगंतुक
mining|mine manager|खान प्रबंधक|submit|जमा|ventilation report|वातायन रिपोर्ट|worker|श्रमिक
sports|federation|महासंघ|issue|जारी|eligibility certificate|पात्रता प्रमाणपत्र|athlete|खिलाड़ी
forestry|forest office|वन कार्यालय|provide|प्रदान|plantation record|रोपण रिकॉर्ड|nursery|पौधशाला
postal|postal operator|डाक संचालक|issue|जारी|delivery receipt|वितरण रसीद|sender|प्रेषक
rail|station manager|स्टेशन प्रबंधक|submit|जमा|platform audit|प्लेटफॉर्म लेखापरीक्षा|traveller|यात्री
disaster|relief agency|राहत संस्था|publish|प्रकाशित|distribution register|वितरण रजिस्टर|recipient|प्राप्तकर्ता
archival|archive|अभिलेखागार|provide|प्रदान|access log|पहुँच रिकॉर्ड|reader|पाठक
fisheries|harbour office|बंदरगाह कार्यालय|issue|जारी|landing receipt|मछली उतराई रसीद|fisher|मछुआरा
broadcast|broadcaster|प्रसारक|submit|जमा|programme log|कार्यक्रम रिकॉर्ड|viewer|दर्शक
laboratory|laboratory|प्रयोगशाला|provide|प्रदान|calibration report|अंशांकन रिपोर्ट|client|ग्राहक'''
def variants(i,a,h,v,w,o,p,b,k):
    n=11+i%9; m=n+15
    past={'publish':'published','provide':'provided','issue':'issued','submit':'submitted'}[v]
    if i%3==0:
        en=f'The {a} shall {v} the {o} within {n} days and retain a copy.'
        hi=f'{h} को {n} दिनों के भीतर {p} {w} करना होगा और एक प्रति रखनी होगी।'
        E=[en.replace('shall','must'),f'The {o} must be {past} by the {a} within {n} days; the {a} must keep a copy.',en.replace(str(n),str(m)),f'The {b}, not the {a}, must {v} the {o} within {n} days and retain a copy.',en.replace(' and retain a copy','')]
        H=[hi.replace('भीतर','अंदर'),f'{p} का {w} {h} द्वारा {n} दिन में किया जाना अनिवार्य है; {h} को उसकी प्रति भी रखनी होगी।',hi.replace(str(n),str(m)),hi.replace(h,k,1),hi.replace(' और एक प्रति रखनी होगी','')]
        labels=['','','altered_numbers','altered_entities','omission']
    elif i%3==1:
        en=f'Only after written approval may the {a} {v} the {o}; no fee may be charged.'
        hi=f'लिखित मंजूरी मिलने के बाद ही {h} {p} {w} कर सकता है; कोई शुल्क नहीं लिया जा सकता।'
        E=[en.replace('written approval','approval in writing'),f'The {a} is permitted to {v} the {o} only once approval is given in writing, and charging a fee is prohibited.',en.replace('Only after written approval','Even without written approval'),f'The {a} must not {v} the {o}, even with written approval; no fee may be charged.',en.split(';')[0]+'.']
        H=[hi.replace('मंजूरी','स्वीकृति'),f'{h} को {p} {w} करने की अनुमति केवल लिखित स्वीकृति के बाद है और शुल्क लेना निषिद्ध है।',hi.replace('लिखित मंजूरी मिलने के बाद ही','लिखित मंजूरी के बिना भी'),f'लिखित मंजूरी मिलने पर भी {h} {p} {w} नहीं कर सकता; शुल्क लेना निषिद्ध है।',hi.split(';')[0]+'।']
        labels=['','','altered_condition','changed_permission','omission']
    else:
        en=f'The {a} must {v} the {o} under section {n}, except during a declared emergency, and notify the {b}.'
        hi=f'घोषित आपातकाल को छोड़कर {h} को धारा {n} के तहत {p} {w} करना होगा और {k} को सूचित करना होगा।'
        E=[en.replace('under section','pursuant to section'),f'Except when an emergency has been declared, the {o} must be {past} by the {a} in accordance with section {n}, and the {b} must be notified by the {a}.',en.replace(f'section {n}',f'section {m}'),f'The {a} must {v} the {o} under section {n} even during a declared emergency and notify the {b}.',en.replace(f', and notify the {b}','')]
        H=[hi.replace('के तहत','के अनुसार'),f'यदि आपातकाल घोषित नहीं है तो धारा {n} के अनुसार {p} का {w} {h} द्वारा किया जाना अनिवार्य है और {k} को सूचना देना भी {h} का दायित्व है।',hi.replace(str(n),str(m)),hi.replace('घोषित आपातकाल को छोड़कर','घोषित आपातकाल के दौरान भी'),hi.replace(f' और {k} को सूचित करना होगा','')]
        labels=['','','altered_reference','missing_exception','omission']
    return [('en',en,E),('hi',hi,H)],labels

def main():
    load=lambda p:pd.read_csv(p,keep_default_na=False)
    base=load('data/contrastive_pilot/train_targeted.csv'); old=load('data/splits/train.csv')
    out=Path('data/v2'); out.mkdir(parents=True,exist_ok=True)
    rows=[[],[]]
    for i,line in enumerate(SPECS.splitlines()):
        domain,*args=line.split('|'); variants_,labels=variants(i,*args); group=f'V2-{i:02d}'
        for lang,orig,edits in variants_:
            for j,(edit,label,score) in enumerate(zip(edits,labels,[9.5,9,2.5,2,6])):
                r=dict.fromkeys(base.columns,''); r.update(pair_id=f'{group}-{lang}-{j}',language=lang,legal_domain=domain,original=orig,simplified=edit,lmp_score=score,category='Preserved' if score>=7.5 else 'Not Preserved' if score<4 else 'Partially Preserved',error_labels=label,source_doc_id=group,group_id=group,split_group=group,concept_id=group,data_origin='ai_generated',dataset_tier='ai_generated',generation_strategy=['near_faithful','rewrite_faithful','near_change','contrast_change','partial_omission'][j],generator='ChatGPT V2 templates',label_source='ai_generation_intent',annotation_status='AI provisional; not independently reviewed',source_provenance='Invented provision',license_note='Project-generated synthetic text'); rows[int(i>=24)].append(r)
    new,challenge=[pd.DataFrame(r,columns=base.columns) for r in rows]
    assert len(new)==240 and len(challenge)==120
    target=pd.concat([base,new],ignore_index=True)
    # Match the total added 360 rows against the original training set, by language/category.
    added=target[~target.pair_id.isin(old.pair_id)]; samples=[]
    for (lang,cat),g in added.groupby(['language','category']):
        s=old[(old.language==lang)&(old.category==cat)].sample(len(g),replace=True,random_state=42).copy(); samples.append(s)
    s=pd.concat(samples,ignore_index=True); s['parent_pair_id']=s.pair_id; s['pair_id']=[f'V2-RESAMPLE-{i}' for i in range(len(s))]; s['generation_strategy']='random_oversampling'
    control=pd.concat([old,s],ignore_index=True)
    assert len(target)==len(control)==748
    ev=[load('data/splits/val.csv'),load('data/splits/test.csv'),load('data/contrastive_pilot/challenge_new.csv'),challenge]
    for t in [target,control]:
        assert t.pair_id.is_unique
        for d in ev:
            for c in ['pair_id','group_id','split_group']: assert not set(t[c])&set(d[c]),c
            assert not set(t.original.map(normalize))&set(d.original.map(normalize))
    for d in [new,challenge]: assert not d.duplicated(['language','original','simplified']).any()
    for name,d in [('train_new',new),('train_targeted',target),('train_resampled',control),('challenge',challenge)]:
        d.to_csv(out/f'{name}.csv',index=False); print(name,len(d)); print(pd.crosstab(d.language,d.category))
    print('Exact isolation checks PASS. Shared template structures and provisional labels remain limitations.')
if __name__=='__main__': main()
