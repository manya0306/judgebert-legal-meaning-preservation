import pytest
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from lmp.config import ROOT

def _app():
    return AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=120).run()

def test_app_runs_and_scores_a_pair():
    """Ported from the original test: the UI changed (the named-example selectbox was replaced by 'Load a real example from the dataset'), the assertions are the same."""
    at = _app(); assert not at.exception
    at.sidebar.checkbox[0].check().run(); at.button[0].click().run()
    assert not at.exception and any("LMP score" in m.label for m in at.metric)

def test_app_scores_custom_hindi_input_and_names_backend():
    at = _app(); at.text_area[0].set_value("किरायेदार मकान मालिक की सहमति के बिना परिसर उप-किराये पर नहीं देगा।").run()
    at.text_area[1].set_value("किरायेदार परिसर उप-किराये पर दे सकता है।").run(); at.button[0].click().run()
    assert not at.exception and {m.label for m in at.metric} == {"LMP score (1-10)", "Category", "Backend"}
    assert any("not a transformer" in w.value.lower() for w in at.warning)      # fallback is explicitly labelled
