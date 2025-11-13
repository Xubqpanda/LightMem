#!/usr/bin/env python3
"""Build one/two/three-hop induced subgraphs for given term(s) from a bipartite export.

Usage (examples):
  # single term
  python scripts/term_subgraph_builder.py --bip /path/to/bip --term "vintage film" --out-dir /path/out --layout radial

  # multiple terms from a file (one per line)
  python scripts/term_subgraph_builder.py --bip /path/to/bip --terms-file /path/terms.txt --out-dir /path/out --layout spring --top-k 40
"""
import argparse
import json
import os
from pathlib import Path
from typing import Set, List, Dict

import networkx as nx
import matplotlib.pyplot as plt


def load_bip(bip_dir: str):
    bip = Path(bip_dir)
    terms = json.loads((bip / 'terms.json').read_text(encoding='utf-8'))
    adj_by_term = json.loads((bip / 'adj_by_term.json').read_text(encoding='utf-8'))
    adj_by_memory = json.loads((bip / 'adj_by_memory.json').read_text(encoding='utf-8'))
    return terms, adj_by_term, adj_by_memory


def find_term_indices(terms: List[str], query_term: str):
    mapping = {t.lower(): i for i, t in enumerate(terms)}
    key = query_term.lower()
    if key in mapping:
        return [mapping[key]]
    # try fuzzy-ish exact: strip punctuation/spaces
    stripped = key.replace(' ', '')
    for t, i in mapping.items():
        if t.replace(' ', '') == stripped:
            return [i]
    return []


def build_one_hop(term_idxs: List[int], adj_by_term: Dict[str, List[str]]):
    mems = set()
    for ti in term_idxs:
        key = str(ti)
        for m in adj_by_term.get(key, []):
            mems.add(m)
    return mems


def build_two_hop(mems: Set[str], adj_by_memory: Dict[str, List[int]]):
    terms = set()
    for m in mems:
        for t in adj_by_memory.get(m, []):
            terms.add(int(t))
    return terms


def build_three_hop(terms: Set[int], adj_by_term: Dict[str, List[str]]):
    mems = set()
    for t in terms:
        key = str(t)
        for m in adj_by_term.get(key, []):
            mems.add(m)
    return mems


def save_subgraph(out_dir: str, name: str, terms_list: List[str], mems: Set[str], edges: List[tuple], layout: str = 'bipartite', top_k: int = 0):
    os.makedirs(out_dir, exist_ok=True)

    g = nx.Graph()
    for t in terms_list:
        g.add_node(t, label=t, bipartite=0)
    memlist = sorted(list(mems))
    for m in memlist:
        display = (m[:8] + '...') if isinstance(m, str) and len(m) > 12 else m
        g.add_node(m, label=display, bipartite=1)
    for a, b in edges:
        g.add_edge(str(a), str(b))

    data = {
        'terms': terms_list,
        'memories': memlist,
        'edges': edges
    }
    with open(os.path.join(out_dir, f'{name}.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Drawing logic (copied/adapted from query_subgraph_builder)
    if layout == 'radial':
        import math
        center_terms = list(terms_list)
        ring1_mems = memlist

        extra_terms = []
        extra_mems = []
        for a, b in edges:
            a = str(a)
            b = str(b)
            if a not in center_terms and a not in memlist and a not in extra_terms:
                extra_terms.append(a)
            if b not in ring1_mems and b not in extra_mems:
                extra_mems.append(b)

        layers = [center_terms, ring1_mems]
        if extra_terms:
            layers.append(extra_terms)
        if extra_mems:
            layers.append(extra_mems)

        pos = {}
        def circle_positions(nodes, radius):
            n = len(nodes)
            if n == 0:
                return {}
            out = {}
            for i, node in enumerate(nodes):
                theta = 2.0 * math.pi * (i / n)
                x = radius * math.cos(theta)
                y = radius * math.sin(theta)
                out[node] = (x, y)
            return out

        for li, nodes in enumerate(layers):
            radius = 0.25 + 0.25 * li
            if li == 0 and len(nodes) == 1:
                pos[nodes[0]] = (0.0, 0.0)
            else:
                pos.update(circle_positions(nodes, radius))

        plt.figure(figsize=(10, 10))
        for li, nodes in enumerate(layers):
            if li % 2 == 0:
                nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color='#1f77b4', node_size=400)
            else:
                nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color='#ff7f0e', node_size=140)

        nx.draw_networkx_edges(g, pos, alpha=0.8, width=1.2, edge_color='#444444')
        labels = {n: (g.nodes[n].get('label') if isinstance(g.nodes[n].get('label'), str) else n) for n in g.nodes()}
        nx.draw_networkx_labels(g, pos, labels=labels, font_size=9)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{name}_radial.png'))
        plt.close()
        return

    # default bipartite
    term_nodes = [n for n in g.nodes() if g.nodes[n].get('bipartite') == 0]
    mem_nodes = [n for n in g.nodes() if g.nodes[n].get('bipartite') == 1]
    def spaced_positions(nodes, x_coord):
        n = len(nodes)
        if n == 0:
            return {}
        ys = list(reversed([i / (n - 1) if n > 1 else 0.5 for i in range(n)]))
        return {node: (x_coord, ys[i]) for i, node in enumerate(nodes)}

    pos = {}
    pos.update(spaced_positions(term_nodes, 0.0))
    pos.update(spaced_positions(mem_nodes, 1.0))

    plt.figure(figsize=(10, max(6, len(term_nodes) * 0.25)))
    nx.draw_networkx_nodes(g, pos, nodelist=term_nodes, node_color='#1f77b4', node_size=400)
    nx.draw_networkx_nodes(g, pos, nodelist=mem_nodes, node_color='#ff7f0e', node_size=140)
    nx.draw_networkx_edges(g, pos, alpha=0.8, width=1.2, edge_color='#444444')
    labels = {n: (g.nodes[n].get('label') if isinstance(g.nodes[n].get('label'), str) else n) for n in g.nodes()}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=9)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{name}.png'))
    plt.close()

    if layout == 'spring':
        if top_k and top_k > 0:
            term_deg = {t: 0 for t in terms_list}
            for a, b in edges:
                a = str(a)
                if a in term_deg:
                    term_deg[a] += 1
            top_terms = set([t for t, _ in sorted(term_deg.items(), key=lambda x: -x[1])[:top_k]])
            seed_terms = set(terms_list)
            filtered_edges = [(a, b) for (a, b) in edges if (a in top_terms or a in seed_terms)]
            nodes_keep = set()
            for a, b in filtered_edges:
                nodes_keep.add(str(a))
                nodes_keep.add(str(b))
            subg = g.subgraph(nodes_keep).copy()
        else:
            subg = g

        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(subg, seed=42, k=0.3, iterations=200)
        term_nodes = [n for n in subg.nodes() if subg.nodes[n].get('bipartite') == 0]
        mem_nodes = [n for n in subg.nodes() if subg.nodes[n].get('bipartite') == 1]
        nx.draw_networkx_nodes(subg, pos, nodelist=term_nodes, node_color='#1f77b4', node_size=300)
        nx.draw_networkx_nodes(subg, pos, nodelist=mem_nodes, node_color='#2ca02c', node_size=80)
        nx.draw_networkx_edges(subg, pos, alpha=0.8, width=1.0, edge_color='#666666')
        labels = {n: (subg.nodes[n].get('label') if isinstance(subg.nodes[n].get('label'), str) else n) for n in subg.nodes()}
        nx.draw_networkx_labels(subg, pos, labels=labels, font_size=8)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{name}_spring.png'))
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bip', required=True, help='Bipartite dir (terms.json, adj_by_term.json, adj_by_memory.json)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--term', help='Single term string to build subgraphs for (quote if contains spaces)')
    group.add_argument('--terms-file', help='File with one term per line')
    parser.add_argument('--out-dir', required=True, help='Parent output dir; will create one subdir per term')
    parser.add_argument('--layout', choices=['bipartite', 'radial', 'spring'], default='spring', help='Layout style for PNG')
    parser.add_argument('--top-k', type=int, default=0, help='When using spring layout, keep top-K terms by degree (0 = all)')
    args = parser.parse_args()

    terms, adj_by_term, adj_by_memory = load_bip(args.bip)

    if args.term:
        query_terms = [args.term]
    else:
        txt = Path(args.terms_file).read_text(encoding='utf-8')
        query_terms = [l.strip() for l in txt.splitlines() if l.strip()]

    parent_out = Path(args.out_dir)
    for qt in query_terms:
        term_idx = find_term_indices(terms, qt)
        if not term_idx:
            print(f"Term not found in bipartite terms.json: '{qt}' — skipping")
            continue

        # build hops (exclusive)
        # 1-hop: memories directly connected to seed terms
        mems_h1 = build_one_hop(term_idx, adj_by_term)
        term_nodes_h1 = [terms[i] for i in term_idx]
        edges_h1 = []
        for ti in term_idx:
            tstr = terms[ti]
            key = str(ti)
            for m in adj_by_term.get(key, []):
                edges_h1.append((tstr, m))

        # 2-hop (memory-focused):
        # - terms_h2: terms connected to mems_h1
        # - mems_h2: memories connected to terms_h2 (this is what we want to display)
        terms_h2 = build_two_hop(mems_h1, adj_by_memory)
        mems_h2 = build_three_hop(terms_h2, adj_by_term)
        terms_h2_list = [terms[i] for i in sorted(list(terms_h2))]
        # edges between terms_h2 and mems_h2 only
        edges_h2 = []
        for ti in terms_h2:
            tstr = terms[ti]
            key = str(ti)
            for m in adj_by_term.get(key, []):
                if m in mems_h2:
                    edges_h2.append((tstr, m))

        # 3-hop (memory-focused): expand one more alternation
        terms_h3 = build_two_hop(mems_h2, adj_by_memory)
        mems_h3 = build_three_hop(terms_h3, adj_by_term)
        terms_h3_list = [terms[i] for i in sorted(list(terms_h3))]
        edges_h3 = []
        for ti in terms_h3:
            tstr = terms[ti]
            key = str(ti)
            for m in adj_by_term.get(key, []):
                if m in mems_h3:
                    edges_h3.append((tstr, m))

        safe_name = qt.replace('/', '_').replace(' ', '_')[:200]
        out_dir = parent_out / safe_name
        out_dir.mkdir(parents=True, exist_ok=True)

        save_subgraph(str(out_dir), 'onehop', term_nodes_h1, mems_h1, edges_h1, layout=args.layout, top_k=args.top_k)
        save_subgraph(str(out_dir), 'twohop', terms_h2_list, mems_h2, edges_h2, layout=args.layout, top_k=args.top_k)
        save_subgraph(str(out_dir), 'threehop', terms_h3_list, mems_h3, edges_h3, layout=args.layout, top_k=args.top_k)

        print(f'Wrote subgraphs for term "{qt}" to {out_dir}')


if __name__ == '__main__':
    main()
