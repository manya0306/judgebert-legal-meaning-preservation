"""Dataset schema. One row = one original-simplified pair (one consolidated record).

Provenance tiers (column `dataset_tier`) are kept strictly apart and are never relabelled as human data:
  original_synthetic : the 48 pairs written by the project author for the first prototype (author-assigned scores)
  ai_generated       : pairs written by an AI model (Claude, in-session) and labelled by that generation pass
  external           : sourced from a real public corpus (none exist in this repository at the time of writing)
  human_annotated    : rows whose labels come from independent human annotation records (none exist yet)
`data_origin` is the older coarse field kept for backward compatibility: synthetic | ai_generated | authentic.
"""
ERROR_LABELS = [
    "omission", "contradiction", "changed_obligation", "changed_permission",
    "altered_condition", "missing_exception", "scope_change",
    "altered_dates", "altered_numbers", "altered_entities", "altered_reference",
]
CATEGORIES = ["Not Preserved", "Partially Preserved", "Preserved"]
DATA_ORIGINS = ["synthetic", "ai_generated", "authentic"]
DATASET_TIERS = ["original_synthetic", "ai_generated", "external", "human_annotated"]
LABEL_SOURCES = ["author_assigned", "ai_generation_intent", "ai_pass2", "ai_adjudicated", "human", "human_adjudicated"]
LANGUAGES = ["en", "hi"]

COLUMNS = [
    "pair_id", "language", "legal_domain", "original", "simplified",
    "lmp_score", "category", "error_labels", "source_doc_id",
    "group_id", "concept_id", "data_origin", "dataset_tier", "generation_strategy", "generator",
    "label_source", "annotation_status", "qa_flags", "source_provenance", "license_note",
]
REQUIRED = ["pair_id", "language", "legal_domain", "original", "simplified", "lmp_score", "source_doc_id", "data_origin"]
