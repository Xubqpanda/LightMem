#!/usr/bin/env python3
"""
Batch query term extractor.

For each question (string) in the input JSON array, extract:
 - NER entities
 - Noun tokens / noun chunks
 - Verb tokens

Combine their lemmas into a deduplicated list "terms" (preserve order), and
write an output JSON array with objects: {"question": ..., "terms": [...]}

Usage:
  python query_entities.py --input-file /path/questions.json --output-dir ./out --model en_core_web_sm
"""
import argparse
import json
import os
import sys
from typing import List, Set

# common stopwords to filter from extracted terms (case-insensitive)
STOPWORDS = set(x.lower() for x in [
    'user','have','they','it','can','that','make','try','use','be','do','i','you',
    # common question words and function words to exclude
    'what', 'when', 'how', 'which', 'who', 'whom', 'whose', 'why',
    'the', 'a', 'an', 'this', 'that', 'these', 'those', 'there', 'here'
])


def load_spacy_model(model_name: str):
    try:
        import spacy
    except Exception as e:
        raise RuntimeError("spaCy is not installed. Install with: pip install spacy") from e

    try:
        nlp = spacy.load(model_name)
    except Exception as e:
        raise RuntimeError(f"spaCy model '{model_name}' not found. Install with: python -m spacy download {model_name}") from e
    return nlp


def extract_terms(nlp, text: str, include_verbs: bool = True) -> List[str]:
    """Extract lemmas from NER entities, noun tokens/chunks, and verb tokens.
    
    This implementation matches the logic in export_collection.py exactly.
    Returns a deduplicated list of lemmas (strings).
    """
    if not text:
        return []
    doc = nlp(text)
    terms: List[str] = []

    # NER
    for ent in doc.ents:
        lemma = " ".join([tok.lemma_ for tok in ent])
        terms.append(lemma.strip())

    # Noun tokens
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN"):
            terms.append(token.lemma_.strip())

    # noun chunks: include lemma-based chunk (captures modifiers but normalized)
    for chunk in doc.noun_chunks:
        # build lemma form of the chunk to normalize plurals/inflections
        lemmas = [tok.lemma_.strip() for tok in chunk if tok.lemma_.strip()]
        if lemmas:
            chunk_lemma = " ".join(lemmas)
            terms.append(chunk_lemma)

    # adjectival modifiers with noun lemma (e.g., 'vintage film')
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN"):
            mods = [m.lemma_.strip() for m in token.lefts if m.dep_ == "amod" and m.pos_ == "ADJ"]
            if mods:
                phrase = " ".join(mods + [token.lemma_.strip()])
                terms.append(phrase)

    # verbs (optional)
    if include_verbs:
        for token in doc:
            if token.pos_ in ("VERB", "AUX"):
                if token.lemma_.lower() in ("be", "do"):
                    continue
                terms.append(token.lemma_.strip())

    # Normalize and deduplicate while preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for t in terms:
        if not t:
            continue
        key = t.lower()
        # filter stoplist
        if key in STOPWORDS:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def process_batch(nlp, questions: List[str], out_dir: str, include_verbs: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    out = []
    total = len(questions)
    for i, q in enumerate(questions, 1):
        terms = extract_terms(nlp, q, include_verbs=include_verbs)
        out.append({"question": q, "terms": terms})
        if i % 50 == 0 or i == total:
            print(f"Processed {i}/{total}")

    out_path = os.path.join(out_dir, "questions_terms.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(out)} items to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True, help='JSON file with questions (array of strings)')
    parser.add_argument('--output-dir', required=True, help='Directory to write outputs')
    parser.add_argument('--model', default='en_core_web_sm', help='spaCy model name')
    parser.add_argument('--exclude-verbs', action='store_true', help='Exclude verbs from extracted terms')
    args = parser.parse_args()

    try:
        nlp = load_spacy_model(args.model)
    except Exception as e:
        print('Error loading spaCy/model:', e, file=sys.stderr)
        sys.exit(2)

    with open(args.input_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    if not isinstance(questions, list):
        print('Input file must be a JSON array of strings', file=sys.stderr)
        sys.exit(1)

    include_verbs = not args.exclude_verbs
    process_batch(nlp, questions, args.output_dir, include_verbs=include_verbs)


if __name__ == '__main__':
    main()