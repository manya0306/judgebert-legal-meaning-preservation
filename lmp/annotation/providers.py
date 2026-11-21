"""Annotation providers. Each returns annotation records (dicts following records.COLUMNS) for a pairs DataFrame.

Honesty rules enforced here:
  * `AnthropicProvider` / `OpenAICompatibleProvider` raise ProviderUnavailable when credentials / a server are missing. They NEVER
    return made-up annotations. (In the delivered repository they were IMPLEMENTED BUT NOT EXECUTED: no key / no network route.)
  * `GenerationIntentProvider` re-expresses the labels the GENERATION pass intended. It is the same author as the data, so it is NOT
    an independent opinion (independence_note says so).
  * `ImportedProvider` loads annotations written by an AI session or a human into the same schema; the caller must state who made them.
  * `HeuristicProvider` is the rule-based analyzer: a non-AI, non-human reference signal, labelled `annotator_type=heuristic`.
"""
from __future__ import annotations
import json, os, re, urllib.request, urllib.error
from datetime import datetime, timezone
import pandas as pd
from ..config import score_to_category
from ..data.schema import ERROR_LABELS

PROMPT_ID = "annotate_lmp_v1"


class ProviderUnavailable(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _expl_from_labels(labels: set[str], score: float) -> str:
    if not labels:
        return "No changed legal element recorded; the simplification keeps the original legal effect." if score >= 7.5 else "Template: no labels recorded."
    return "Template (not free text): changed legal element(s) recorded at generation time - " + ", ".join(sorted(labels)) + "."


class GenerationIntentProvider:
    annotator_id = "claude-gen-intent"
    def annotate(self, pairs: pd.DataFrame) -> list[dict]:
        out = []
        for r in pairs.itertuples():
            labs = {x for x in str(r.error_labels or "").split("|") if x}
            out.append({"pair_id": r.pair_id, "annotator_id": self.annotator_id, "annotator_type": "ai", "provider": "in_session",
                        "model": "claude-sonnet-5.5", "model_version": "", "pass_id": "pass1_generation", "timestamp_utc": "",
                        "language": r.language, "legal_domain": r.legal_domain, "lmp_score": r.lmp_score,
                        "category": score_to_category(r.lmp_score), "error_labels": "|".join(sorted(labs)), "changed_elements": "[]",
                        "explanation": _expl_from_labels(labs, r.lmp_score), "evidence_original": "", "evidence_simplified": "",
                        "confidence": "", "review_status": "unreviewed", "prompt_id": "generation_time",
                        "independence_note": "Same author as the text; labels were decided while writing the variant. NOT independent."})
        return out


class HeuristicProvider:
    """Rule-based analyzer turned into a (weak) annotator: score from restored findings. Non-AI, non-human reference signal."""
    annotator_id = "rule-analyzer-v2"
    def annotate(self, pairs: pd.DataFrame) -> list[dict]:
        from ..explain.analyzer import analyze
        out = []
        for r in pairs.itertuples():
            a = analyze(r.original, r.simplified, r.language)
            weight = sum(f.confidence for f in a.findings)
            score = 10.0 if not a.findings else max(1.0, 9.0 - 2.5 * weight)
            labs = sorted(a.labels)
            out.append({"pair_id": r.pair_id, "annotator_id": self.annotator_id, "annotator_type": "heuristic", "provider": "local_rules",
                        "model": "lmp.explain.analyzer", "model_version": "v2", "pass_id": "heuristic", "timestamp_utc": _now(),
                        "language": r.language, "legal_domain": r.legal_domain, "lmp_score": round(score, 1),
                        "category": score_to_category(round(score, 1)), "error_labels": "|".join(labs),
                        "changed_elements": json.dumps([f.to_dict() for f in a.findings][:6], ensure_ascii=False),
                        "explanation": "; ".join(f.detected_change for f in a.findings)[:300], "evidence_original": "", "evidence_simplified": "",
                        "confidence": "", "review_status": "unreviewed", "prompt_id": "n/a",
                        "independence_note": "Deterministic rules; shares lexicon with the explainer. Not a semantic judgement."})
        return out


class ImportedProvider:
    """Load annotations produced elsewhere (an AI session transcript exported to JSONL/CSV, or human sheets)."""
    def __init__(self, path):
        self.path = path
    def annotate(self, pairs: pd.DataFrame) -> list[dict]:
        from .records import load_annotations
        ann = load_annotations(self.path)
        return ann[ann.pair_id.isin(set(pairs.pair_id))].to_dict("records")


def _build_prompt(r) -> str:
    from pathlib import Path
    from ..config import ROOT
    t = (ROOT / "data/annotation/prompts/annotate_lmp_v1.md").read_text(encoding="utf-8")
    t = t.split("Language:", 1)[1] if "Language:" in t else t
    return "Language:" + t.replace("{language}", r.language).replace("{legal_domain}", r.legal_domain).replace("{original}", r.original).replace("{simplified}", r.simplified)


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in model reply")
    return json.loads(m.group(0))


def _to_record(r, obj: dict, annotator_id, provider, model, version, pass_id, note) -> dict:
    labs = [l for l in obj.get("error_labels", []) if l in ERROR_LABELS]
    sc = float(obj["lmp_score"])
    return {"pair_id": r.pair_id, "annotator_id": annotator_id, "annotator_type": "ai", "provider": provider, "model": model,
            "model_version": version, "pass_id": pass_id, "timestamp_utc": _now(), "language": r.language, "legal_domain": r.legal_domain,
            "lmp_score": sc, "category": score_to_category(sc), "error_labels": "|".join(labs),
            "changed_elements": json.dumps(obj.get("changed_elements", []), ensure_ascii=False), "explanation": obj.get("explanation", ""),
            "evidence_original": obj.get("evidence_original", ""), "evidence_simplified": obj.get("evidence_simplified", ""),
            "confidence": obj.get("confidence", ""), "review_status": "unreviewed", "prompt_id": PROMPT_ID, "independence_note": note}


class AnthropicProvider:
    """Anthropic Messages API via urllib (no extra dependency). Requires ANTHROPIC_API_KEY. NOT EXECUTED in the delivered repo."""
    def __init__(self, model: str, annotator_id: str | None = None, temperature: float = 0.0, api_key: str | None = None, max_tokens: int = 600):
        self.key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise ProviderUnavailable("ANTHROPIC_API_KEY is not set - cannot call the Anthropic API")
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.annotator_id = annotator_id or f"anthropic:{model}"
    def annotate(self, pairs: pd.DataFrame) -> list[dict]:
        out = []
        for r in pairs.itertuples():
            body = json.dumps({"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                               "messages": [{"role": "user", "content": _build_prompt(r)}]}).encode()
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST",
                                         headers={"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    reply = json.loads(resp.read())
            except (urllib.error.URLError, TimeoutError) as e:
                raise ProviderUnavailable(f"API call failed: {e}") from e
            text = "".join(b.get("text", "") for b in reply.get("content", []))
            out.append(_to_record(r, _parse_json(text), self.annotator_id, "anthropic_api", reply.get("model", self.model), reply.get("model", ""),
                                  "api_pass", f"API model call, temperature={self.temperature}; independent of other models only if the model differs"))
        return out


class OpenAICompatibleProvider:
    """Any OpenAI-compatible chat endpoint (Ollama, llama.cpp server, vLLM, ...). Requires a reachable base_url. NOT EXECUTED here."""
    def __init__(self, base_url: str, model: str, annotator_id: str | None = None, api_key: str = "none", temperature: float = 0.0):
        self.base_url, self.model, self.key, self.temperature = base_url.rstrip("/"), model, api_key, temperature
        self.annotator_id = annotator_id or f"local:{model}"
        try:
            urllib.request.urlopen(self.base_url + "/models", timeout=3)
        except Exception as e:
            raise ProviderUnavailable(f"No OpenAI-compatible server at {base_url}: {e}") from e
    def annotate(self, pairs: pd.DataFrame) -> list[dict]:
        out = []
        for r in pairs.itertuples():
            body = json.dumps({"model": self.model, "temperature": self.temperature,
                               "messages": [{"role": "user", "content": _build_prompt(r)}]}).encode()
            req = urllib.request.Request(self.base_url + "/chat/completions", data=body, method="POST",
                                         headers={"Authorization": f"Bearer {self.key}", "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                reply = json.loads(resp.read())
            text = reply["choices"][0]["message"]["content"]
            out.append(_to_record(r, _parse_json(text), self.annotator_id, "openai_compatible", self.model, "", "local_pass",
                                  "Local model; independent of Claude passes (different model family)"))
        return out


PROVIDERS = {"generation_intent": GenerationIntentProvider, "heuristic": HeuristicProvider, "imported": ImportedProvider,
             "anthropic": AnthropicProvider, "openai_compatible": OpenAICompatibleProvider}
