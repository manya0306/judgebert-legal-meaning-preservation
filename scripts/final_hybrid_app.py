from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from lmp.models.final_hybrid import MODEL,evidence
from lmp.explain.analyzer import analyze
from lmp.models.legal_equivalence import equivalence
st.title('Multilingual legal meaning preservation')
st.caption('Research prototype — not legal advice. Scores and categories may disagree.')
@st.cache_resource
def resources():
    return joblib.load('outputs/final_hybrid/selected.joblib'),SentenceTransformer(MODEL,device='cpu',local_files_only=True)
lang=st.selectbox('Language',['en','hi']); a=st.text_area('Original provision'); b=st.text_area('Simplified provision')
if st.button('Assess') and a.strip() and b.strip():
    bundle,enc=resources(); cache=dict(zip([a,b],enc.encode([a,b],normalize_embeddings=True)))
    d=pd.DataFrame([dict(original=a,simplified=b,language=lang)]); score,cat,prob=bundle['model'].predict(d,cache)
    findings = [f.to_dict() for f in analyze(a,b,lang).findings]+evidence(a,b,lang)
    reason = equivalence(a,b,lang)
    st.caption(f"Raw learned model: {score[0]:.2f}/10 — {cat[0]}")
    if reason:
        st.metric('LMP score','10.00/10')
        st.write('Classification: Preserved')
        st.info('Deterministic equivalence rule: '+reason+'. This is not a learned prediction.')
    else:
        st.metric('LMP score',f'{score[0]:.2f}/10')
        st.write('Classification:',cat[0])
        expected = 'Not Preserved' if score[0]<4 else ('Partially Preserved' if score[0]<7.5 else 'Preserved')
        if expected != cat[0]: st.warning('Score and classification disagree; manual review recommended.')
    st.caption(f"Training: {bundle['training']}; features: {bundle['mode']}")
    st.write('Potential legal changes')
    if findings: st.json(findings)
    else: st.write('No changes detected by the heuristic explainer; this does not prove equivalence.')
