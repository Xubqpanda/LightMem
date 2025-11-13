#!/usr/bin/env python3
"""Apply conservative filtering:
- remove verb tokens from `questions_terms.json` (preserve multiword phrases that contain no verb tokens)
- apply conservative_stoplist.txt to entries' `terms` (case-insensitive exact matches)

Outputs:
- exports/questions_terms_conservative_no_verbs.json
- exports/e47becba_entries_filtered_conservative.json
"""
from pathlib import Path
import json
import sys

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "dataset" / "longmemeval" / "out_entities"
EXPORTS = BASE / "pre_experiments" / "exports"

try:
    import spacy
except Exception as e:
    print("spaCy not available; please install with: pip install spacy && python -m spacy download en_core_web_sm", file=sys.stderr)
    raise


def load_stoplist(path: Path):
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    return {line.strip().lower() for line in text.splitlines() if line.strip()}


def remove_verbs_from_questions(nlp, questions_path: Path, out_path: Path):
    data = json.loads(questions_path.read_text(encoding="utf-8"))
    out = []
    for item in data:
        terms = item.get("terms", [])
        kept = []
        for term in terms:
            # multiword terms: tokenise and check POS of each token; if any VERB/AUX -> drop
            doc = nlp(term)
            has_verb = any(tok.pos_ in {"VERB", "AUX"} for tok in doc)
            if not has_verb:
                kept.append(term)
        new_item = dict(item)
        new_item["terms"] = kept
        out.append(new_item)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(out)} questions) ")


def apply_stoplist_to_entries(entries_path: Path, stoplist: set, out_path: Path):
    data = json.loads(entries_path.read_text(encoding="utf-8"))
    changed = 0
    out = []
    for item in data:
        terms = item.get("terms", [])
        new_terms = [t for t in terms if t.lower() not in stoplist]
        if len(new_terms) != len(terms):
            changed += 1
        new_item = dict(item)
        new_item["terms"] = new_terms
        out.append(new_item)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(out)} entries). Entries with changes: {changed}")


def main():
    nlp = spacy.load("en_core_web_sm")
    stoplist_path = EXPORTS / "conservative_stoplist.txt"
    stoplist = load_stoplist(stoplist_path)

    questions_in = DATA / "questions_terms.json"
    questions_out = EXPORTS / "questions_terms_conservative_no_verbs.json"
    remove_verbs_from_questions(nlp, questions_in, questions_out)

    entries_in = EXPORTS / "e47becba_entries_filtered_strong_findoff.json"
    entries_out = EXPORTS / "e47becba_entries_filtered_conservative.json"
    apply_stoplist_to_entries(entries_in, stoplist, entries_out)


if __name__ == "__main__":
    main()
