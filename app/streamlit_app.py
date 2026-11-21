"""Streamlit demo for Explainable Multilingual Legal Meaning Preservation (research prototype). Run:  streamlit run app/streamlit_app.py
Every result comes from lmp.inference.LMPPredictor (the real pipeline). The backend in use is always displayed."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import pandas as pd, streamlit as st
from lmp.inference import LMPPredictor, InferenceError
from lmp.explain.highlight import highlight
from lmp.config import load_config
from lmp.data.io import load_pairs

st.set_page_config(page_title="Legal Meaning Preservation", layout="wide")

@st.cache_resource(show_spinner="Loading model ...")
def get_predictor(prefer: str, ckpt: str):
    return LMPPredictor(checkpoint_dir=ckpt or None, prefer=prefer)

@st.cache_data
def get_examples():
    cfg = load_config(); df = load_pairs(ROOT / cfg["data_path"]); return df[df.dataset_tier.isin(["ai_generated", "original_synthetic"])].reset_index(drop=True)

CSS_REMOVED, CSS_ADDED = "background:#ffb3b3;padding:0 2px;border-radius:3px", "background:#b8e6b8;padding:0 2px;border-radius:3px"

st.title("Explainable Multilingual Legal Meaning Preservation")
st.caption("Research prototype inspired by JUDGEBERT. Not legal advice. Scores come from a model trained on a small AI-assisted dataset; explanations are rule-based heuristics, not validated legal reasoning.")
with st.sidebar:
    st.header("Settings")
    prefer = st.selectbox("Model", ["auto", "transformer", "lexrule_fallback", "tfidf"], help="'auto' uses a trained transformer checkpoint if one exists, otherwise the labelled fallback.")
    ckpt = st.text_input("Transformer checkpoint dir (optional)", "")
    mode = {"lexrule_fallback": "auto", "auto": "auto", "transformer": "transformer", "tfidf": "tfidf"}[prefer]
    lang = st.selectbox("Language", ["auto-detect", "en", "hi"])
    min_conf = st.slider("Minimum rule confidence to display", 0.0, 1.0, 0.5, 0.05)
    use_ex = st.checkbox("Load a real example from the dataset")
try:
    pred = get_predictor(mode if prefer != "lexrule_fallback" else "auto", ckpt if prefer != "lexrule_fallback" else "/nonexistent")
except InferenceError as e:
    st.error(str(e)); st.stop()
st.sidebar.info(f"**Active backend:** {pred.backend}\n\n{pred.describe()}")
orig_default = simp_default = ""; ref = None
if use_ex:
    ex = get_examples(); i = st.sidebar.number_input("Example index", 0, len(ex) - 1, 0); r = ex.iloc[int(i)]; orig_default, simp_default, ref = r.original, r.simplified, r
c1, c2 = st.columns(2)
original = c1.text_area("Original legal sentence", orig_default, height=140, key=f"o{orig_default[:20]}")
simplified = c2.text_area("Simplified sentence", simp_default, height=140, key=f"s{simp_default[:20]}")
if ref is not None:
    st.caption(f"Dataset row `{ref.pair_id}` ({ref.language}, {ref.legal_domain}); reference label **{ref.lmp_score:g} / {ref.category}** from `{ref.label_source}` - AI-assisted/author-assigned, NOT human-annotated. The model has not necessarily been held out from this row.")
if st.button("Analyse", type="primary"):
    try:
        res = pred.predict(original, simplified, None if lang == "auto-detect" else lang)
    except InferenceError as e:
        st.error(str(e)); st.stop()
    m1, m2, m3 = st.columns(3); m1.metric("LMP score (1-10)", f"{res.score:.1f}"); m2.metric("Category", res.category); m3.metric("Backend", res.backend)
    if res.class_probs: st.write({k: round(v, 3) for k, v in res.class_probs.items()})
    for w in res.warnings: st.warning(w)
    fs = [f for f in res.analysis.findings if f.confidence >= min_conf]
    st.subheader("Potentially altered legal elements (rule-based, heuristic)")
    if not fs: st.info(res.analysis.no_findings_note or "No finding above the confidence threshold.")
    o_sp = [s for f in fs for s in f.orig_spans]; s_sp = [s for f in fs for s in f.simp_spans]
    h1, h2 = st.columns(2); h1.markdown("**Original** (flagged text)", help="Highlights come from the rules, not from the neural model."); h1.markdown(highlight(original, o_sp), unsafe_allow_html=True)
    h2.markdown("**Simplified** (flagged text)"); h2.markdown(highlight(simplified, s_sp), unsafe_allow_html=True)
    for f in fs:
        with st.expander(f"{f.category} - {f.element}  |  confidence {f.confidence:.2f} ({f.uncertainty} uncertainty)  |  layer: {f.layer}"):
            st.write(f"**Detected textual change:** {f.detected_change}"); st.write(f"**Possible legal relevance (heuristic interpretation):** {f.interpretation}")
            st.write(f"Original: `{f.original_expression}` -> Simplified: `{f.simplified_expression}`"); st.caption(f"rule `{f.rule}`; heuristic, not validated by a legal expert")
    with st.expander("Textual differences (surface alignment, not a legal conclusion)"):
        st.dataframe(pd.DataFrame([{"op": d["op"], "original": d["original"], "simplified": d["simplified"]} for d in res.analysis.diff]), width="stretch")
    st.caption(res.analysis.disclaimer)
