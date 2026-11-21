import random
import string
import pandas as pd

from lmp.data.splits import document_split, _group_ids


def _distinct_text(key: str) -> str:
    """Pseudo-sentence unique to `key` (so near-duplicate grouping does not merge unrelated fixture rows)."""
    rnd = random.Random(key)
    return " ".join("".join(rnd.choice(string.ascii_lowercase) for _ in range(rnd.randint(4, 9))) for _ in range(9))


def test_bilingual_templates_never_cross_splits():
    rows = []

    for domain in ["EMP", "TEN", "INS"]:
        for template in ["A", "B", "C", "D"]:
            for language in ["EN", "HI"]:
                doc = f"SYN-{language}-{domain}-{template}"
                rows.append({
                    "pair_id": f"{doc}-001",
                    "source_doc_id": doc,
                    "language": language.lower(),
                    "legal_domain": domain.lower(),
                    "lmp_score": 9.0,
                    "original": _distinct_text(f"{domain}{template}"),
                    "simplified": _distinct_text(f"s{domain}{template}"),
                })

    df = pd.DataFrame(rows)
    train, val, test = document_split(df, seed=42)

    train_groups = set(_group_ids(train))
    val_groups = set(_group_ids(val))
    test_groups = set(_group_ids(test))

    assert not (train_groups & val_groups)
    assert not (train_groups & test_groups)
    assert not (val_groups & test_groups)


def test_every_pair_appears_in_exactly_one_split():
    df = pd.DataFrame([
        {
            "pair_id": f"SYN-{lang}-{domain}-{template}-001",
            "source_doc_id": f"SYN-{lang}-{domain}-{template}",
            "language": lang.lower(),
            "legal_domain": domain.lower(),
            "lmp_score": 9.0,
            "original": _distinct_text(f"{domain}{template}"),
            "simplified": _distinct_text(f"s{domain}{template}"),
        }
        for domain in ["EMP", "TEN", "INS"]
        for template in ["A", "B", "C", "D"]
        for lang in ["EN", "HI"]
    ])

    train, val, test = document_split(df, seed=42)

    all_ids = (
        train["pair_id"].tolist()
        + val["pair_id"].tolist()
        + test["pair_id"].tolist()
    )

    assert len(all_ids) == len(df)
    assert len(set(all_ids)) == len(df)
