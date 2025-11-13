#!/usr/bin/env python3
"""Suggest a conservative stoplist from a bipartite export using spaCy + statistics.

Outputs:
 - exports/conservative_stoplist_suggested.txt  (one term per line)
 - exports/conservative_stoplist_candidates.csv (detailed metrics for review)

Heuristics used (configurable):
 - classify a term as 'noun_only' if all tokens are NOUN/PROPN or punctuation
 - degree = number of linked memories
 - specificity = log(total_mems / (1 + degree))
 - suggestion rules (example):
     * if not noun_only and degree >= verb_degree_thresh -> suggest
     * if noun_only and degree >= noun_degree_thresh and specificity <= spec_thresh -> suggest

This script only suggests; it does NOT automatically modify existing stoplists.
"""
from pathlib import Path
import json
import math
import csv
import argparse
import sys


def load_bip(bip_dir: Path):
    terms = json.loads((bip_dir / 'terms.json').read_text(encoding='utf-8'))
    adj_by_term = json.loads((bip_dir / 'adj_by_term.json').read_text(encoding='utf-8'))
    adj_by_memory = json.loads((bip_dir / 'adj_by_memory.json').read_text(encoding='utf-8'))
    return terms, adj_by_term, adj_by_memory


def load_spacy(model_name: str):
    try:
        import spacy
    except Exception as e:
        raise RuntimeError('spaCy is required. Install with: pip install spacy') from e
    try:
        nlp = spacy.load(model_name)
    except Exception as e:
        raise RuntimeError(f"spaCy model '{model_name}' not found. Install with: python -m spacy download {model_name}") from e
    return nlp


def term_pos_info(nlp, term: str):
    # If term is empty return conservative defaults
    if not term or not term.strip():
        return {'tokens': [], 'pos_tags': [], 'is_noun_only': False}
    doc = nlp(term)
    pos_tags = [tok.pos_ for tok in doc if not tok.is_space]
    # consider token sets that are allowed for noun-only: NOUN, PROPN, ADJ, DET, PUNCT
    allowed = {'NOUN', 'PROPN', 'ADJ', 'DET', 'PUNCT'}
    is_noun_only = all((p in allowed) for p in pos_tags) and len(pos_tags) > 0
    return {'tokens': [tok.text for tok in doc], 'pos_tags': pos_tags, 'is_noun_only': is_noun_only}


def suggest_stoplist(bip_dir: str, out_dir: str, model: str, verb_degree_thresh: int = 20, noun_degree_thresh: int = 50, spec_thresh: float = 1.0, top_k: int = 500):
    bip = Path(bip_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    terms, adj_by_term, adj_by_memory = load_bip(bip)
    nlp = load_spacy(model)

    total_mems = len(adj_by_memory)

    rows = []
    for i, t in enumerate(terms):
        key = str(i)
        degree = len(adj_by_term.get(key, []))
        posinfo = term_pos_info(nlp, t)
        is_noun_only = posinfo['is_noun_only']
        specificity = math.log((total_mems / (1 + degree))) if degree >= 0 else 0.0

        suggested = False
        reason = ''
        if not is_noun_only and degree >= verb_degree_thresh:
            suggested = True
            reason = f'non-noun and degree>={verb_degree_thresh}'
        elif is_noun_only and degree >= noun_degree_thresh and specificity <= spec_thresh:
            suggested = True
            reason = f'noun high-degree and low-specificity'

        rows.append({
            'idx': i,
            'term': t,
            'degree': degree,
            'is_noun_only': is_noun_only,
            'pos_tags': ','.join(posinfo['pos_tags']),
            'specificity': round(specificity, 4),
            'suggested': suggested,
            'reason': reason,
        })

    # sort suggested candidates by degree desc then specificity asc
    suggested_rows = [r for r in rows if r['suggested']]
    suggested_sorted = sorted(suggested_rows, key=lambda r: (-r['degree'], r['specificity']))

    # write CSV of candidates (top_k rows by degree)
    csv_path = out / 'conservative_stoplist_candidates.csv'
    with open(csv_path, 'w', encoding='utf-8', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=['idx', 'term', 'degree', 'is_noun_only', 'pos_tags', 'specificity', 'suggested', 'reason'])
        writer.writeheader()
        for r in sorted(rows, key=lambda r: (-r['degree'], r['specificity']))[:top_k]:
            writer.writerow(r)

    # write suggested stoplist
    stoplist_path = out / 'conservative_stoplist_suggested.txt'
    with open(stoplist_path, 'w', encoding='utf-8') as sf:
        for r in suggested_sorted:
            sf.write(r['term'] + "\n")

    print(f'Wrote candidate CSV: {csv_path} ({len(rows)} terms total)')
    print(f'Wrote suggested stoplist: {stoplist_path} ({len(suggested_sorted)} suggested)')

    # print top 50 suggestions for quick review
    print('\nTop suggestions (preview):')
    for i, r in enumerate(suggested_sorted[:50], start=1):
        print(f"{i:2d}. degree={r['degree']:3d} spec={r['specificity']:5.2f} noun_only={r['is_noun_only']} idx={r['idx']:4d} term={r['term']} reason={r['reason']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bip', required=True, help='bipartite export dir (terms.json, adj_by_term.json, adj_by_memory.json)')
    parser.add_argument('--out-dir', default='exports', help='output dir for candidates/stoplist')
    parser.add_argument('--model', default='en_core_web_sm', help='spaCy model name (default: en_core_web_sm)')
    parser.add_argument('--verb-degree-thresh', type=int, default=20)
    parser.add_argument('--noun-degree-thresh', type=int, default=50)
    parser.add_argument('--spec-thresh', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=500)
    args = parser.parse_args()

    suggest_stoplist(args.bip, args.out_dir, args.model, args.verb_degree_thresh, args.noun_degree_thresh, args.spec_thresh, args.top_k)


if __name__ == '__main__':
    main()
