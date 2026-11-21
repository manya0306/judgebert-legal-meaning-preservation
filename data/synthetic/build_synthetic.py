"""Builds data/synthetic/synthetic_pairs.csv.

IMPORTANT: Every row is SYNTHETIC. The sentences were written by the project author for software testing
and demonstration. They are NOT extracted from real legal documents, and the LMP scores are the author's
own assigned values, NOT human annotations from independent annotators. The dataset (48 pairs) is far too small for
research-grade training or for drawing conclusions about model quality.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd

D = "|".join
# (domain, doc_letter, original, simplified, author_score, error_labels)
EN = [
 ("tenancy","A","The tenant shall give 30 days' written notice before vacating.","The tenant may tell the landlord before leaving.",2,"changed_obligation|altered_numbers|omission"),
 ("tenancy","A","The landlord shall return the security deposit within 15 days of the end of the tenancy.","The landlord will return the deposit after the tenancy ends.",4,"altered_numbers|omission"),
 ("tenancy","B","The tenant shall not sublet the premises without the landlord's written consent.","The tenant must get the landlord's written permission before subletting the premises.",9,""),
 ("tenancy","B","Rent must be paid on or before the 5th day of each month.","Rent must be paid on or before the 10th day of each month.",3,"altered_dates"),
 ("tenancy","C","If rent is not paid on time, the landlord may terminate the agreement.","The landlord may terminate the agreement.",3,"altered_condition|omission"),
 ("tenancy","C","The tenant shall not make any permanent alterations to the premises.","The tenant may make some changes to the premises.",1,"contradiction|changed_permission"),
 ("tenancy","D","The landlord shall bear the cost of repairs unless the damage was caused by the tenant.","The landlord shall bear the cost of repairs.",3,"missing_exception"),
 ("tenancy","D","The tenant and members of the tenant's family may reside in the premises.","Only the tenant may live in the premises.",3,"scope_change|altered_entities"),
 ("employment","A","The employee shall give two months' notice before resigning.","The employee has to tell the employer two months before resigning.",9,""),
 ("employment","A","The employee may take up to 12 days of casual leave per year.","The employee may take casual leave during the year.",5,"altered_numbers|omission"),
 ("employment","B","The employer shall pay salary on or before the 7th of each month.","The employer shall pay salary on or before the 17th of each month.",3,"altered_dates"),
 ("employment","B","The employee shall not disclose confidential information during or after employment.","The employee shall not disclose confidential information during employment.",3,"scope_change|omission"),
 ("employment","C","Overtime must be paid at twice the normal rate.","Overtime must be paid at one and a half times the normal rate.",3,"altered_numbers"),
 ("employment","C","The employer may terminate the contract without notice in cases of gross misconduct.","The employer may terminate the contract without notice.",3,"altered_condition|omission"),
 ("employment","D","The employee must complete a probation period of six months before confirmation.","New employees will be confirmed after a probation period of six months.",9,""),
 ("employment","D","The employee shall not work for a competitor for one year after leaving the company.","The employee may work for a competitor after leaving the company.",1,"contradiction|changed_obligation|altered_numbers"),
 ("insurance","A","The insurer shall pay the claim within 30 days of receiving all required documents.","The insurer will pay the claim within 30 days of receiving the required documents.",9,""),
 ("insurance","A","The policyholder shall notify the insurer of any loss within 7 days.","The policyholder should inform the insurer of a loss soon.",3,"changed_obligation|altered_numbers"),
 ("insurance","B","Damage caused by flood is not covered under this policy.","Damage caused by flood is covered under this policy.",1,"contradiction"),
 ("insurance","B","The insurer shall not be liable for losses arising from intentional acts of the policyholder.","The insurer is not responsible for losses the policyholder causes on purpose.",9,""),
 ("insurance","C","Coverage applies only to the property located at the address stated in the policy schedule.","Coverage applies to the policyholder's properties.",2,"scope_change|omission"),
 ("insurance","C","The policyholder may cancel the policy at any time by giving written notice.","The policyholder can cancel the policy at any time.",6,"omission|altered_condition"),
 ("insurance","D","A deductible of Rs. 5,000 shall apply to each claim.","A deductible of Rs. 50,000 shall apply to each claim.",2,"altered_numbers"),
 ("insurance","D","The policy expires on 31 March 2026 unless renewed.","The policy expires on 31 March 2026.",4,"missing_exception"),
]
HI = [
 ("tenancy","A","किरायेदार को घर खाली करने से पहले 30 दिन का लिखित नोटिस देना होगा।","किरायेदार घर छोड़ने से पहले मकान मालिक को बता सकता है।",2,"changed_obligation|altered_numbers|omission"),
 ("tenancy","A","मकान मालिक सुरक्षा जमा राशि 15 दिनों के भीतर लौटाएगा।","मकान मालिक सुरक्षा जमा राशि कुछ दिनों में लौटाएगा।",4,"altered_numbers"),
 ("tenancy","B","किरायेदार मकान मालिक की लिखित अनुमति के बिना मकान किसी और को किराये पर नहीं दे सकता।","किरायेदार को मकान किसी और को किराये पर देने के लिए मकान मालिक की लिखित अनुमति लेनी होगी।",9,""),
 ("tenancy","B","किराया हर महीने की 5 तारीख तक चुकाना होगा।","किराया हर महीने की 10 तारीख तक चुकाना होगा।",3,"altered_dates"),
 ("tenancy","C","यदि किराया समय पर नहीं चुकाया गया, तो मकान मालिक अनुबंध समाप्त कर सकता है।","मकान मालिक अनुबंध समाप्त कर सकता है।",3,"altered_condition|omission"),
 ("tenancy","C","किरायेदार मकान में किसी भी प्रकार का स्थायी बदलाव नहीं करेगा।","किरायेदार मकान में कुछ बदलाव कर सकता है।",1,"contradiction|changed_permission"),
 ("tenancy","D","मरम्मत का खर्च मकान मालिक वहन करेगा, जब तक कि क्षति किरायेदार के कारण न हुई हो।","मरम्मत का खर्च मकान मालिक वहन करेगा।",3,"missing_exception"),
 ("tenancy","D","किरायेदार और उसके परिवार के सदस्य मकान में रह सकते हैं।","केवल किरायेदार मकान में रह सकता है।",3,"scope_change|altered_entities"),
 ("employment","A","कर्मचारी को इस्तीफा देने से दो महीने पहले नोटिस देना होगा।","कर्मचारी को इस्तीफे से दो महीने पहले नियोक्ता को बताना होगा।",9,""),
 ("employment","A","कर्मचारी प्रति वर्ष अधिकतम 12 दिन की आकस्मिक छुट्टी ले सकता है।","कर्मचारी वर्ष में आकस्मिक छुट्टी ले सकता है।",5,"altered_numbers|omission"),
 ("employment","B","नियोक्ता हर महीने की 7 तारीख तक वेतन का भुगतान करेगा।","नियोक्ता हर महीने की 17 तारीख तक वेतन का भुगतान करेगा।",3,"altered_dates"),
 ("employment","B","कर्मचारी नौकरी के दौरान और उसके बाद भी गोपनीय जानकारी का खुलासा नहीं करेगा।","कर्मचारी नौकरी के दौरान गोपनीय जानकारी का खुलासा नहीं करेगा।",3,"scope_change|omission"),
 ("employment","C","ओवरटाइम का भुगतान सामान्य दर से दोगुनी दर पर किया जाना चाहिए।","ओवरटाइम का भुगतान सामान्य दर से डेढ़ गुना दर पर किया जाना चाहिए।",3,"altered_numbers"),
 ("employment","C","घोर कदाचार की स्थिति में नियोक्ता बिना नोटिस दिए अनुबंध समाप्त कर सकता है।","नियोक्ता बिना नोटिस दिए अनुबंध समाप्त कर सकता है।",3,"altered_condition|omission"),
 ("employment","D","कर्मचारी को स्थायी होने से पहले छह महीने की परिवीक्षा अवधि पूरी करनी होगी।","नए कर्मचारी छह महीने की परिवीक्षा अवधि के बाद स्थायी किए जाएंगे।",9,""),
 ("employment","D","कर्मचारी कंपनी छोड़ने के बाद एक वर्ष तक किसी प्रतिस्पर्धी के लिए काम नहीं करेगा।","कर्मचारी कंपनी छोड़ने के बाद किसी प्रतिस्पर्धी के लिए काम कर सकता है।",1,"contradiction|changed_obligation|altered_numbers"),
 ("insurance","A","बीमाकर्ता सभी आवश्यक दस्तावेज़ प्राप्त होने के 30 दिनों के भीतर दावे का भुगतान करेगा।","बीमाकर्ता आवश्यक दस्तावेज़ मिलने के 30 दिनों के भीतर दावे का भुगतान करेगा।",9,""),
 ("insurance","A","पॉलिसीधारक को किसी भी हानि की सूचना 7 दिनों के भीतर बीमाकर्ता को देनी होगी।","पॉलिसीधारक को हानि की सूचना जल्दी बीमाकर्ता को देनी चाहिए।",3,"changed_obligation|altered_numbers"),
 ("insurance","B","बाढ़ से हुई क्षति इस पॉलिसी के अंतर्गत कवर नहीं है।","बाढ़ से हुई क्षति इस पॉलिसी के अंतर्गत कवर है।",1,"contradiction"),
 ("insurance","B","बीमाकर्ता पॉलिसीधारक के जानबूझकर किए गए कार्यों से हुई हानि के लिए उत्तरदायी नहीं होगा।","पॉलिसीधारक जो हानि जानबूझकर करता है, उसके लिए बीमाकर्ता उत्तरदायी नहीं है।",9,""),
 ("insurance","C","कवरेज केवल पॉलिसी अनुसूची में दिए गए पते की संपत्ति पर लागू होगा।","कवरेज पॉलिसीधारक की संपत्तियों पर लागू होगा।",2,"scope_change|omission"),
 ("insurance","C","पॉलिसीधारक लिखित सूचना देकर किसी भी समय पॉलिसी रद्द कर सकता है।","पॉलिसीधारक किसी भी समय पॉलिसी रद्द कर सकता है।",6,"omission|altered_condition"),
 ("insurance","D","प्रत्येक दावे पर 5,000 रुपये की कटौती लागू होगी।","प्रत्येक दावे पर 50,000 रुपये की कटौती लागू होगी।",2,"altered_numbers"),
 ("insurance","D","यह पॉलिसी 31 मार्च 2026 को समाप्त हो जाएगी, जब तक कि इसका नवीनीकरण न किया जाए।","यह पॉलिसी 31 मार्च 2026 को समाप्त हो जाएगी।",4,"missing_exception"),
]

def build() -> pd.DataFrame:
    rows = []
    for lang, data in (("en", EN), ("hi", HI)):
        for i, (dom, doc, o, s, score, labs) in enumerate(data, 1):
            rows.append({
                "pair_id": f"SYN-{lang.upper()}-{i:03d}", "language": lang, "legal_domain": dom,
                "original": o, "simplified": s, "lmp_score": float(score), "category": "",
                "error_labels": labs, "source_doc_id": f"SYN-{lang.upper()}-{dom[:3].upper()}-{doc}",
                "data_origin": "synthetic",
                "source_provenance": "Written by the project author for software testing; not from any real legal document",
                "license_note": "Project-original text; no third-party rights",
                "annotation_status": "author_assigned_score_not_human_annotation",
            })
    df = pd.DataFrame(rows)
    from lmp.config import score_to_category
    df["category"] = df["lmp_score"].map(score_to_category)
    return df

if __name__ == "__main__":
    out = Path(__file__).with_name("synthetic_pairs.csv")
    df = build()
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"wrote {len(df)} synthetic pairs -> {out}")
    print(df.groupby(["language", "legal_domain"]).size())
