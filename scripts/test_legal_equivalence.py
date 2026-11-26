from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lmp.models.legal_equivalence import equivalence
base='The tenant shall pay rent within 30 days.'
cases=[(base,base,'en',True),(base,base.replace('shall','must'),'en',True),(base,base.replace('shall','may'),'en',False),(base,base.replace('30','60'),'en',False),(base,base.replace('tenant','landlord'),'en',False),(base,base.replace('shall','must not'),'en',False),('The tenant shall not pay rent.','The tenant must not pay rent.','en',False),('कर्मचारी को आवेदन करना होगा।','कर्मचारी को आवेदन करना होगा।','hi',True),('कर्मचारी को 30 दिनों में आवेदन करना होगा।','कर्मचारी को 60 दिनों में आवेदन करना होगा।','hi',False)]
for a,b,lang,expected in cases:
    assert bool(equivalence(a,b,lang))==expected,(a,b)
print(f'{len(cases)} equivalence tests PASS')
