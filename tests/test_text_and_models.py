import numpy as np
from lmp.langid import identify_language
from lmp.preprocess import normalize, tokenize
from lmp.models.tfidf import TfidfBaseline
from lmp.evaluation.metrics import regression_metrics, category_metrics, label_detection_metrics
from lmp.config import score_to_category

def test_langid():
    assert identify_language("The tenant shall pay rent.") == "en"
    assert identify_language("किरायेदार को किराया देना होगा।") == "hi"
    assert identify_language("1234 !!") == "unknown"

def test_normalize_and_hindi_tokens_stay_whole():
    assert normalize("  The  tenant’s\n duty ") == "The tenant's duty"
    assert "किरायेदार" in tokenize("किरायेदार को घर खाली करना होगा।")

def test_tfidf_identical_unrelated():
    t = TfidfBaseline().fit(["the tenant shall pay rent", "insurer pays claim within 30 days", "कर्मचारी वेतन प्राप्त करेगा"])
    assert t.predict(["the tenant shall pay rent"], ["the tenant shall pay rent"])[0] > 9.99
    assert t.predict(["the tenant shall pay rent"], ["insurer pays claim within 30 days"])[0] < 2.5

def test_tfidf_misses_legal_change_which_motivates_the_project():
    o, s = "The tenant shall give 30 days' written notice before vacating.", "The tenant may give 30 days' written notice before vacating."
    t = TfidfBaseline().fit([o, s, "other text about insurance policy"])
    assert t.predict([o], [s])[0] > 5      # lexically similar although obligation became permission

def test_metrics():
    m = regression_metrics([1, 5, 9, 3], [1.5, 5, 8, 3.5]); assert m["rmse"] > 0 and m["pearson"] > 0.9
    assert score_to_category(2) == "Not Preserved" and score_to_category(5) == "Partially Preserved" and score_to_category(9) == "Preserved"
    assert category_metrics([2, 5, 9], [2, 5, 9])["kappa"] == 1.0
    r = label_detection_metrics([{"omission"}, set()], [{"omission"}, {"omission"}]); assert r["omission"]["precision"] == 0.5 and r["omission"]["recall"] == 1.0
