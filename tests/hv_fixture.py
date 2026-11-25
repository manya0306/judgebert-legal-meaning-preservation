"""Synthetic fixtures for the human-validation analysis tests: tiny workbooks with the REAL layouts
(banner row 1 + header row 2 for annotation sheets, header row 1 for the key) and a miniature data/splits tree."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook

from lmp.human_validation import ingest as I

N_EN, N_HI, L_EN, L_HI = 8, 6, 3, 3
SPEC = I.DesignSpec(n_en=N_EN, n_hi=N_HI, lawyer_en=L_EN, lawyer_hi=L_HI, expected_student_split=None, expected_expert_split=None,
                    corrections=(("HV2-HI-001", 8, 6, "Partially Preserved"), ("HV2-EN-002", 7, 6, "Partially Preserved")))
DOMAINS = ["tenancy", "employment", "contract", "consumer"]
COMPLEX = ["easy", "moderate", "difficult"]


def _cat(s):
    return I.rubric_category(s)


def _repo_cat(s):
    return "Not Preserved" if s < 4 else "Preserved" if s >= 7.5 else "Partially Preserved"


class Fixture:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for lang, n in (("en", N_EN), ("hi", N_HI)):
            for k in range(1, n + 1):
                t = float(rng.integers(1, 11))
                split = "test" if k <= (2 if lang == "en" else 3) else ("train" if k <= (7 if lang == "en" else 5) else "validation")
                grp = f"G-{split[:2]}-{(k + 1) // 2:03d}"
                text = f"The tenant shall pay {k} days rent to landlord {k}." if lang == "en" else f"किरायेदार {k} दिन में किराया देगा।"
                rows.append(dict(neutral=f"HV2-{lang.upper()}-{k:03d}", source=f"AIG-C{(k + 1) // 2:03d}-{lang.upper()}-{k:02d}", language=lang, split=split, group_id=grp,
                                 concept_id=grp.replace("G-", "C-"), legal_domain=DOMAINS[k % 4], complexity=COMPLEX[k % 3], original=text, simplified=text.replace("shall pay", "pays") if lang == "en" else text.replace("देगा", "देता है"),
                                 truth=t, ref=float(np.clip(round((t + rng.normal(0.5, 1.0)) * 2) / 2, 1, 10))))
        self.pairs = pd.DataFrame(rows)
        self.spec = SPEC
        self.rng = rng
        self._build()

    def _build(self):
        rng, p = self.rng, self.pairs
        src = pd.DataFrame({"pair_id": p.source, "language": p.language, "legal_domain": p.legal_domain, "original": p.original, "simplified": p.simplified,
                            "lmp_score": p.ref, "category": p.ref.map(_repo_cat), "error_labels": np.where(p.ref < 7, "altered_numbers|omission", ""), "source_doc_id": "D-" + p.group_id,
                            "group_id": p.group_id, "concept_id": p.concept_id, "split_group": p.group_id, "data_origin": "ai_generated", "dataset_tier": "ai_generated",
                            "label_source": "ai_generation_intent", "annotation_status": "ai_single_pass_unreviewed", "dataset_split_source": p.split})
        self.src = src
        key = pd.DataFrame({"pair_id": p.neutral, "source_pair_id": p.source, "language": p.language, "legal_domain": p.legal_domain, "complexity_level": p.complexity,
                            "dataset_split": p.split, "is_heldout_test": p.split == "test", "selection_source": np.where(p.split == "test", "held-out TEST split (primary)", "SUPPLEMENTAL"),
                            "original": p.original, "simplified": p.simplified, "AI_REFERENCE_score": p.ref, "AI_REFERENCE_category": p.ref.map(_repo_cat),
                            "AI_REFERENCE_error_labels": src.error_labels.values, "reference_label_type": "AI-GENERATED: single pass by in-session Claude at generation time; unreviewed; NOT human",
                            "label_source": "ai_generation_intent", "annotation_status": "ai_single_pass_unreviewed", "group_id": p.group_id, "concept_id": p.concept_id,
                            "source_doc_id": "D-" + p.group_id, "data_origin": "ai_generated", "in_lawyer_packet": False})
        for c in I.STAT_COLS:
            key[c] = 1
        for lang, n in (("en", L_EN), ("hi", L_HI)):
            key.loc[key.pair_id.isin([f"HV2-{lang.upper()}-{k:03d}" for k in range(1, n + 1)]), "in_lawyer_packet"] = True
        self.key = key
        assign = []
        for a, langs in SPEC.annotators:
            ids = key[key.language.isin(langs)].pair_id.tolist()
            assign += [dict(annotator=a, position=i + 1, pair_id=x, source_pair_id=key.set_index("pair_id").at[x, "source_pair_id"], language=key.set_index("pair_id").at[x, "language"]) for i, x in enumerate(ids)]
        self.assign = pd.DataFrame(assign)
        self.student = {}
        for a, langs in SPEC.annotators:
            rows = []
            for r in p[p.language.isin(langs)].sample(frac=1, random_state=hash(a) % 1000).itertuples():
                s = int(np.clip(round(r.truth + rng.normal(0, 1.2)), 1, 10))
                rows.append({I.S_COL["pair_id"]: r.neutral, I.S_COL["language"]: "English" if r.language == "en" else "Hindi", I.S_COL["domain"]: r.legal_domain.title(),
                             I.S_COL["complexity"]: r.complexity.title(), I.S_COL["original"]: r.original, I.S_COL["simplified"]: r.simplified, I.S_COL["score"]: s,
                             I.S_COL["category"]: _cat(s), I.S_COL["confidence"]: str(rng.choice(I.CONFIDENCES)),
                             I.S_COL["change"]: "None (preserved)" if s >= 7 else str(rng.choice(I.CHANGE_TYPES[1:6])), I.S_COL["explanation"]: f"{a} explanation for {r.neutral}", I.S_COL["notes"]: None})
            self.student[a] = pd.DataFrame(rows)
        rows = []
        for r in p[key.in_lawyer_packet.values].itertuples():
            s = {"HV2-HI-001": 6, "HV2-EN-002": 6}.get(r.neutral, int(np.clip(round(r.truth + rng.normal(0, 1)), 1, 10)))
            rows.append({I.L_COL["pair_id"]: r.neutral, I.L_COL["language"]: "English" if r.language == "en" else "Hindi", I.L_COL["domain"]: r.legal_domain.title(),
                         I.L_COL["complexity"]: r.complexity.title(), I.L_COL["original"]: r.original, I.L_COL["simplified"]: r.simplified, I.L_COL["score"]: s,
                         I.L_COL["category"]: _cat(s), I.L_COL["change"]: "None (preserved)" if s >= 7 else "Condition", I.L_COL["explanation"]: f"legal reasoning {r.neutral}",
                         I.L_COL["confidence"]: "Low", I.L_COL["notes"]: None, I.L_COL["comments"]: None})
        self.lawyer = pd.DataFrame(rows)

    # ---- writing
    def _sheet(self, wb, name, df, banner=True):
        ws = wb.create_sheet(name)
        if banner:
            ws.append(["banner text"])
        ws.append(list(df.columns) + ["Consistency check (auto)"])
        for r in df.itertuples(index=False):
            ws.append([None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v) for v in r])

    def write(self, root: Path):
        d = Path(root)
        (d / "completed").mkdir(parents=True, exist_ok=True)
        (d / "data" / "splits").mkdir(parents=True, exist_ok=True)
        wb = Workbook()
        wb.active.title = "Instructions"
        for a, df in self.student.items():
            self._sheet(wb, a, df)
        wb.save(d / "completed" / I.STUDENT_FILE)
        wb = Workbook()
        wb.active.title = "Guidelines"
        self._sheet(wb, "Lawyer_Review", self.lawyer)
        wb.save(d / "completed" / I.LAWYER_FILE)
        wb = Workbook()
        wb.active.title = "README_PRIVATE"
        ws = wb.create_sheet("Master_Key")
        ws.append(list(self.key.columns))
        for r in self.key.itertuples(index=False):
            ws.append([v.item() if hasattr(v, "item") else v for v in r])
        ws = wb.create_sheet("Assignment_Map")
        ws.append(list(self.assign.columns))
        for r in self.assign.itertuples(index=False):
            ws.append(list(r))
        wb.save(d / "completed" / I.KEY_FILE)
        for fname, split in I.SPLIT_FILES:
            self.src[self.src.dataset_split_source == split].drop(columns="dataset_split_source").to_csv(d / "data" / "splits" / f"{fname}.csv", index=False)
        return d
