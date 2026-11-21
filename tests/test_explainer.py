import pytest
from lmp.explain.analyzer import analyze, extract_features

def cats(o, s, lang): return {f.category for f in analyze(o, s, lang).findings}

def test_project_example_english():
    c = cats("The tenant shall give 30 days' written notice before vacating.", "The tenant may tell the landlord before leaving.", "en")
    assert {"changed_obligation", "altered_numbers", "omission"} <= c

def test_project_example_hindi():
    a = analyze("किरायेदार को घर खाली करने से पहले 30 दिन का लिखित नोटिस देना होगा।", "किरायेदार घर छोड़ने से पहले मकान मालिक को बता सकता है।", "hi")
    assert {"changed_obligation", "altered_numbers"} <= a.labels
    f = [x for x in a.findings if x.category == "changed_obligation"][0]
    assert "होगा" in f.original_expression and "सकता" in f.simplified_expression

def test_faithful_paraphrase_not_flagged_for_modality():
    c = cats("The tenant shall not sublet the premises without the landlord's written consent.", "The tenant must get the landlord's written permission before subletting the premises.", "en")
    assert not c & {"contradiction", "changed_obligation", "changed_permission", "omission"}

def test_identical_pairs_have_no_findings():
    for o, l in (("The employer shall pay salary by the 7th of each month.", "en"), ("नियोक्ता हर महीने की 7 तारीख तक वेतन का भुगतान करेगा।", "hi")):
        assert analyze(o, o, l).findings == []

def test_negation_flip_and_prohibition_to_permission():
    assert "contradiction" in cats("Damage caused by flood is not covered under this policy.", "Damage caused by flood is covered under this policy.", "en")
    assert "contradiction" in cats("कर्मचारी कंपनी छोड़ने के बाद एक वर्ष तक किसी प्रतिस्पर्धी के लिए काम नहीं करेगा।", "कर्मचारी कंपनी छोड़ने के बाद किसी प्रतिस्पर्धी के लिए काम कर सकता है।", "hi")

def test_conditions_and_exceptions():
    assert "altered_condition" in cats("If rent is not paid on time, the landlord may terminate the agreement.", "The landlord may terminate the agreement.", "en")
    assert "missing_exception" in cats("The policy expires on 31 March 2026 unless renewed.", "The policy expires on 31 March 2026.", "en")
    assert "missing_exception" in cats("मरम्मत का खर्च मकान मालिक वहन करेगा, जब तक कि क्षति किरायेदार के कारण न हुई हो।", "मरम्मत का खर्च मकान मालिक वहन करेगा।", "hi")

def test_negation_inside_conditional_clause_is_not_a_polarity_flip():
    assert "contradiction" not in cats("यदि किराया समय पर नहीं चुकाया गया, तो मकान मालिक अनुबंध समाप्त कर सकता है।", "मकान मालिक अनुबंध समाप्त कर सकता है।", "hi")

def test_numbers_dates_and_modal_may_is_not_a_month():
    assert "altered_numbers" in cats("A deductible of Rs. 5,000 shall apply.", "A deductible of Rs. 50,000 shall apply.", "en")
    assert "altered_dates" in cats("Rent must be paid on or before the 5th day of each month.", "Rent must be paid on or before the 10th day of each month.", "en")
    assert extract_features("The tenant may leave.", "en")["dates"] == []
    assert "altered_numbers" in cats("ओवरटाइम का भुगतान सामान्य दर से दोगुनी दर पर किया जाना चाहिए।", "ओवरटाइम का भुगतान सामान्य दर से डेढ़ गुना दर पर किया जाना चाहिए।", "hi")

def test_output_separates_textual_change_from_interpretation_and_has_disclaimer():
    a = analyze("The tenant shall pay.", "The tenant may pay.", "en")
    f = a.findings[0]; assert f.detected_change and f.interpretation and f.detected_change != f.interpretation and 0 < f.confidence <= 1
    assert "not legal advice" in a.disclaimer

def test_optional_model_signal_uses_supplied_scorer():
    scorer = lambda o, s: [5.0 + 0.1 * len(x) for x in s]
    a = analyze("The tenant shall give 30 days' notice.", "The tenant may give notice.", "en", scorer=scorer)
    assert any(f.model_signal is not None for f in a.findings)
