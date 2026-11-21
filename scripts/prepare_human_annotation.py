
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "splits" / "test.csv"
OUT = ROOT / "outputs" / "human_validation"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261121
rng = np.random.default_rng(SEED)

df = pd.read_csv(DATA).fillna("")
assert len(df) == 112
assert df["pair_id"].is_unique

# Allocate 25 examples proportionally across the existing categories.
def category_quotas(frame, n=25):
    counts = frame["category"].value_counts()
    exact = counts / counts.sum() * n
    quotas = np.floor(exact).astype(int)
    remaining = n - quotas.sum()

    remainders = (exact - quotas).sort_values(ascending=False)
    for category in remainders.index[:remaining]:
        quotas[category] += 1

    return quotas.to_dict()


# Prefer examples from underrepresented groups, domains and
# annotation sources, while maintaining category quotas.
def select_language(frame, n=25):
    quotas = category_quotas(frame, n)

    chosen = []
    group_counts = {}
    domain_counts = {}
    source_counts = {}

    for category, quota in quotas.items():
        candidates = frame[frame["category"] == category].copy()
        candidates["_random"] = rng.random(len(candidates))

        for _ in range(quota):
            if candidates.empty:
                raise ValueError(
                    f"Not enough examples for category: {category}"
                )

            def penalty(row):
                return (
                    4 * group_counts.get(row["group_id"], 0)
                    + 2 * domain_counts.get(row["legal_domain"], 0)
                    + source_counts.get(row["label_source"], 0)
                )

            candidates["_penalty"] = candidates.apply(penalty, axis=1)
            candidates = candidates.sort_values(
                ["_penalty", "_random"]
            )

            selected = candidates.iloc[0]
            chosen.append(selected.drop(
                labels=["_random", "_penalty"]
            ).to_dict())

            group = selected["group_id"]
            domain = selected["legal_domain"]
            source = selected["label_source"]

            group_counts[group] = group_counts.get(group, 0) + 1
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1

            candidates = candidates[
                candidates["pair_id"] != selected["pair_id"]
            ]

    result = pd.DataFrame(chosen)
    return result.sample(frac=1, random_state=SEED).reset_index(drop=True)


# Select independently within each language.
english = select_language(df[df["language"] == "en"].copy())
hindi = select_language(df[df["language"] == "hi"].copy())

selected = pd.concat([english, hindi], ignore_index=True)

assert len(english) == 25
assert len(hindi) == 25
assert selected["pair_id"].is_unique

# Assignment: all four review English; first three review Hindi.
assignments = {
    "A1": selected,
    "A2": selected,
    "A3": selected,
    "A4": selected[selected["language"] == "en"].copy()
}

# Category labels are derived from scores for human annotations.
instructions = [
    ["Purpose", "Compare the original and simplified sentence for preservation of legal meaning."],
    ["Independence", "Do not discuss examples or compare answers before all annotations are submitted."],
    ["Blinding", "Do not look up AI labels, scores, model predictions or other annotators' responses."],
    ["Score 8–10", "Preserved: material legal meaning is retained."],
    ["Score 4–7", "Partially Preserved: some material legal element is changed, weakened, strengthened, omitted or ambiguous."],
    ["Score 1–3", "Not Preserved: central legal meaning is materially reversed, contradicted or removed."],
    ["Meaning changed", "Choose Yes, No or Unclear."],
    ["Change types", "Use labels such as changed_obligation, changed_permission, contradiction, altered_condition, missing_exception, scope_change, altered_dates, altered_numbers, altered_entities, omission, or none."],
    ["Rationale", "Explain briefly what changed or why meaning was preserved. Point to relevant words where possible."],
    ["Confidence", "High, Medium or Low."],
    ["Hindi", "Assess the Hindi sentence directly. Do not translate mechanically word-for-word."],
    ["Important", "These are synthetic/AI-assisted research examples, not verified clauses from real legal instruments."]
]

for annotator, packet in assignments.items():
    packet = packet.copy()
    packet = packet.sample(frac=1, random_state=SEED)

    wb = Workbook()
    info = wb.active
    info.title = "Instructions"

    for row in instructions:
        info.append(row)

    ws = wb.create_sheet("Annotations")
    headers = [
        "pair_id", "language", "original", "simplified",
        "score_1_to_10", "category", "meaning_changed",
        "change_types", "rationale", "confidence", "complete"
    ]
    ws.append(headers)

    for _, r in packet.iterrows():
        ws.append([
            r["pair_id"], r["language"],
            r["original"], r["simplified"],
            None, None, None, None, None, None, "No"
        ])

    # Dropdowns and validation.
    score = DataValidation(
        type="whole", operator="between",
        formula1="1", formula2="10"
    )
    category = DataValidation(
        type="list",
        formula1='"Not Preserved,Partially Preserved,Preserved"'
    )
    changed = DataValidation(
        type="list", formula1='"Yes,No,Unclear"'
    )
    confidence = DataValidation(
        type="list", formula1='"High,Medium,Low"'
    )
    complete = DataValidation(type="list", formula1='"Yes,No"')

    for dv in [score, category, changed, confidence, complete]:
        ws.add_data_validation(dv)

    last = ws.max_row
    score.add(f"E2:E{last}")
    category.add(f"F2:F{last}")
    changed.add(f"G2:G{last}")
    confidence.add(f"J2:J{last}")
    complete.add(f"K2:K{last}")

    for sheet in wb.worksheets:
        sheet.freeze_panes = "A2"
        sheet.sheet_view.showGridLines = False

        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F4E78")
            cell.alignment = Alignment(wrap_text=True)

        for row in sheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(
                    wrap_text=True, vertical="top"
                )

    widths = [16, 12, 65, 65, 18, 24, 18, 38, 55, 16, 14]
    for i, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = width

    info.column_dimensions["A"].width = 22
    info.column_dimensions["B"].width = 105

    path = OUT / f"Annotation_Packet_{annotator}.xlsx"
    wb.save(path)
    print("Created:", path)

# Separate private key: never distribute this to annotators.
private_columns = [
    "pair_id", "language", "original", "simplified",
    "lmp_score", "category", "error_labels",
    "legal_domain", "group_id", "split_group",
    "label_source", "data_origin", "dataset_tier",
    "source_provenance"
]

private = selected[private_columns].copy()
private["sampling_seed"] = SEED

private_path = OUT / "PRIVATE_Reference_Key.xlsx"
private.to_excel(private_path, index=False)

print("\nSelection summary:")
print(selected.groupby(["language", "category"]).size())
print("\nLegal domain coverage:")
print(selected.groupby(["language", "legal_domain"]).size())
print("\nSelected pairs:", len(selected))
print("Private key:", private_path)
