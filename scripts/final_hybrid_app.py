from pathlib import Path
import sys
import json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import streamlit as st
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from lmp.models.final_hybrid import MODEL, evidence
from lmp.explain.analyzer import analyze
from lmp.models.legal_equivalence import equivalence
from improve_explainer import LABELS, features, rules

st.set_page_config(page_title='Legal Meaning Preservation', layout='wide')
st.title('Multilingual Legal Meaning Preservation')
st.caption('English / Hindi • Research prototype • Not legal advice')

@st.cache_resource
def resources():
    bundle = joblib.load(ROOT / 'outputs/final_hybrid/selected.joblib')
    if bundle['training'] != 'targeted' or bundle['mode'] != 'hybrid':
        raise ValueError('Expected V1 targeted hybrid checkpoint.')
    return bundle, SentenceTransformer(MODEL, device='cpu', local_files_only=True)

@st.cache_resource
def verifier_resources():
    v = joblib.load(ROOT / 'outputs/explainer_verifier/verifier.joblib')
    if v['labels'] != LABELS:
        raise ValueError('Verifier label mismatch')
    config = json.loads((ROOT / 'outputs/explainer_final/filter_config.json').read_text())
    return v, float(config['threshold'])

with st.sidebar:
    st.header('Model and limitations')
    st.write('V1 targeted hybrid: frozen multilingual MiniLM, lexical and directional legal features, gradient-boosted prediction heads.')
    st.write('The explanation engine is heuristic. Verifier agreement is not proof of legal correctness. Human evaluation is limited and Hindi performance is weaker.')
    st.write('No finding does not establish equivalence. All consequential decisions require professional review.')
lang = st.selectbox('Language', ['en','hi'], format_func=lambda x: {'en':'English','hi':'Hindi'}[x])
with st.form('assessment'):
    c1,c2 = st.columns(2)
    a = c1.text_area('Original provision', height=180)
    b = c2.text_area('Simplified provision', height=180)
    submitted = st.form_submit_button('Assess meaning preservation')
if submitted:
    if not a.strip() or not b.strip():
        st.warning('Enter both provisions.')
        st.stop()
    try:
        with st.spinner('Assessing provisions...'):
            bundle, enc = resources()
            cache = dict(zip([a,b], enc.encode([a,b], normalize_embeddings=True)))
            d = pd.DataFrame([dict(original=a,simplified=b,language=lang)])
            score,cat,prob = bundle['model'].predict(d,cache)
            findings = [f.to_dict() for f in analyze(a,b,lang).findings] + evidence(a,b,lang)
            reason = equivalence(a,b,lang)
    except Exception as exc:
        st.error(f'Assessment failed: {exc}')
        st.stop()
    raw = float(score[0]); category = str(cat[0]); confidence = float(max(prob[0]))
    st.subheader('Assessment')
    c1,c2,c3 = st.columns(3)
    c1.metric('System score', f'{10 if reason else raw:.2f}/10')
    c2.metric('System category', 'Preserved' if reason else category)
    c3.metric('Raw model max probability', f'{confidence:.1%}')
    st.write(f'Raw learned prediction: {raw:.2f}/10 — {category}')
    if reason:
        st.info(f'Deterministic equivalence adjustment: {reason}. Not a learned prediction or legal guarantee.')
    expected = 'Not Preserved' if raw < 4 else ('Partially Preserved' if raw < 7.5 else 'Preserved')
    if expected != category:
        st.warning(f'Raw regression category ({expected}) and classification head disagree. Manual review recommended.')
    if confidence < .6:
        st.warning('Low model probability under the provisional 0.60 review threshold; probabilities are not calibrated.')
    with st.expander('Raw classification probabilities'):
        st.dataframe(pd.DataFrame({'Category':bundle['model'].clf.classes_, 'Probability':prob[0]}))
    st.subheader('Potential legal changes')
    scores = {}; threshold = None
    try:
        v, threshold = verifier_resources()
        x = features(d,v['vectorizer'],rules(d))
        scores = {k:float(m.predict_proba(x)[0,1]) for k,m in zip(LABELS,v['models']) if m is not None}
    except Exception as exc:
        st.warning(f'Optional verifier unavailable; all rule findings retained. ({exc})')
    if not findings:
        st.info('No changes detected by the heuristic explainer. This does not establish equivalence.')
    else:
        def supported(f):
            return threshold is not None and f.get('category') in scores and scores[f['category']] >= threshold
        for f in sorted(findings,key=lambda f:not supported(f)):
            k = f.get('category','unknown')
            p = scores.get(k)
            status = 'Rule + verifier agreement' if supported(f) else ('Rule only — below verifier threshold' if p is not None else 'Rule only — verifier unavailable for label')
            with st.expander(k.replace('_',' ').title() + ' | ' + status, expanded=True):
                if p is not None:
                    st.caption(f'Uncalibrated verifier probability: {p:.2f}; prioritization threshold: {threshold:.2f}')
                left,right = st.columns(2)
                left.write('**Original expression**'); left.write(str(f.get('original_expression','Not available')))
                right.write('**Simplified expression**'); right.write(str(f.get('simplified_expression','Not available')))
                st.write(str(f.get('interpretation','Potential change; verify in context.')))
                if f.get('detected_change'): st.write(str(f['detected_change']))
        st.caption('All rule findings are retained. Agreement only changes display order; no verifier-only claims are added.')
    st.warning('Research output only. Review the complete provisions and applicable legal context before relying on any result.')
