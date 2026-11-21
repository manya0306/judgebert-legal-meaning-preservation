import pathlib
import pytest
from lmp.explain.analyzer import analyze
from lmp.explain.highlight import highlight
from lmp.explain.spans import word_diff
from lmp.inference import LMPPredictor, InferenceError

def cats(o, s, l="en"): return {f.category for f in analyze(o, s, l).findings}

def test_negation_and_modal_and_number_changes():
    assert "contradiction" in cats("The tenant shall pay rent.", "The tenant shall not pay rent.")
    assert cats("The landlord shall give notice.", "The landlord may give notice.") & {"changed_obligation", "changed_permission"}
    assert "altered_numbers" in cats("Notice of 24 hours is required.", "Notice of 48 hours is required.")

def test_and_or_flip_comparator_reference_and_party_order():
    assert "altered_condition" in cats("No licence shall be cancelled unless the holder has been heard and the board consented.", "No licence shall be cancelled unless the holder has been heard or the board consented.")
    assert "contradiction" in cats("The price shall be lower than the cost.", "The price shall be higher than the cost.")
    assert "altered_reference" in cats("An order under section 14 shall not take effect.", "An order under section 41 shall not take effect.")
    assert "altered_entities" in cats("The tenant shall not sublet without the consent of the landlord.", "The landlord shall not sublet without the consent of the tenant.")

def test_hindi_findings_and_reference():
    assert "altered_reference" in cats("धारा 14 के अंतर्गत आदेश प्रभावी नहीं होगा।", "धारा 41 के अंतर्गत आदेश प्रभावी नहीं होगा।", "hi")
    assert "altered_numbers" in cats("किरायेदार को 24 घंटे की सूचना देनी होगी।", "किरायेदार को 48 घंटे की सूचना देनी होगी।", "hi")
    assert cats("मकान मालिक को सूचना देनी होगी।", "मकान मालिक सूचना दे सकता है।", "hi")

def test_identical_sentences_give_no_findings_and_unrelated_is_not_called_preserved_by_rules():
    a = analyze("The tenant shall pay rent.", "The tenant shall pay rent."); assert a.findings == [] and "does NOT show" in a.no_findings_note
    assert analyze("The tenant shall pay rent monthly.", "Copyright lasts sixty years after death.").diff

def test_spans_are_valid_offsets_and_layers_labelled():
    o, s = "The tenant shall pay 24 hours rent.", "The tenant may pay 48 hours rent."
    for f in analyze(o, s).findings:
        assert f.layer == "rule_based" and f.uncertainty in ("low", "medium", "high")
        assert all(0 <= a < b <= len(o) for a, b in f.orig_spans) and all(0 <= a < b <= len(s) for a, b in f.simp_spans)
    d = word_diff(o, s); assert d and all(x["op"] in ("replace", "delete", "insert") for x in d)

def test_highlight_escapes_html():
    h = highlight("a <b> c", [(2, 5)]); assert "&lt;b&gt;" in h and "<b>" not in h.replace("<mark", "")

@pytest.fixture(scope="module")
def pred(): return LMPPredictor(checkpoint_dir="checkpoints/does_not_exist")

def test_fallback_is_labelled_not_transformer(pred):
    r = pred.predict("The tenant shall pay rent monthly.", "The tenant pays rent every month.")
    assert r.backend == "lexrule_gbm" and not r.is_transformer and 1 <= r.score <= 10 and any("not a transformer" in w.lower() for w in r.warnings)

def test_inference_hindi_and_english_and_autodetect(pred):
    assert pred.predict("किरायेदार किराया देगा।", "किरायेदार हर महीने किराया देगा।").language == "hi"
    assert pred.predict("The tenant shall pay rent.", "The tenant pays rent.", "auto").language == "en"

def test_inference_edge_cases(pred):
    with pytest.raises(InferenceError): pred.predict("", "x y z")
    with pytest.raises(InferenceError): pred.predict("a b c d", "a b c", "xx")
    assert pred.predict("The tenant shall pay rent.", "The tenant shall pay rent.").score == 10.0
    r = pred.predict("The tenant shall pay rent monthly.", "Copyright subsists for sixty years after the death of the author."); assert 1 <= r.score <= 10

def test_missing_checkpoint_explicit_transformer_request_errors():
    with pytest.raises(InferenceError): LMPPredictor(checkpoint_dir="checkpoints/none", prefer="transformer")

def test_streamlit_app_smoke():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(pathlib.Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"), default_timeout=120).run(); assert not at.exception
    at.sidebar.checkbox[0].check().run(); at.button[0].click().run(); assert not at.exception and len(at.metric) == 3
    at.text_area[0].set_value("").run(); at.button[0].click().run(); assert at.error
