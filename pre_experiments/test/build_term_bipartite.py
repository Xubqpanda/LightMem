#!/usr/bin/env python3
"""Build a term-memory bipartite graph with multi-hop visualization.

Multi-hop definition: Each hop ENDS at a memory node.
- Hop 1: Query Term → Memory1
- Hop 2: Memory1 → Term → Memory2  
- Hop 3: Memory2 → Term → Memory3

Usage: 
  python build_term_bipartite.py --entries PATH --out-dir PATH \
    --query-terms "machine learning,neural network" --max-hops 3 --draw-multihop
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter, deque

def load_entries(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_mappings(entries):
    adj_by_term = defaultdict(set)
    adj_by_memory = defaultdict(set)
    for rec in entries:
        mid = rec.get('id')
        terms = rec.get('terms') or []
        for t in terms:
            key = t.strip()
            if not key:
                continue
            adj_by_term[key].add(mid)
            adj_by_memory[mid].add(key)
    return adj_by_term, adj_by_memory

def write_json(obj, path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def multi_hop_bfs(query_terms, adj_by_term, adj_by_memory, max_hops=3,
                   topic_map=None, topic_id=None, filter_each_hop=False):
    """BFS to find multi-hop neighbors. Each hop MUST end at a memory node.
    
    We stop expansion when we've reached max_hops memories.
    Intermediate terms that don't lead to the next hop memory are excluded.
    """
    memory_hops = {}  # memory_node -> hop distance
    term_hops = {}    # term_node -> hop distance
    edges = []
    
    queue = deque()
    visited = set()
    
    # Initialize with query terms at hop 0
    for qt in query_terms:
        if qt in adj_by_term:
            term_node = f'T:{qt}'
            term_hops[term_node] = 0
            visited.add(term_node)
            queue.append((term_node, 0))
    
    while queue:
        node, last_mem_hop = queue.popleft()
        
        if node.startswith('T:'):
            # Term → Memory (increment hop)
            term = node[2:]
            next_hop = last_mem_hop + 1
            
            if next_hop > max_hops:
                # This term doesn't lead anywhere within max_hops
                # Remove it from term_hops if it was added
                continue
            
            # Try to reach memories
            reached_any_memory = False
            for mem in adj_by_term.get(term, []):
                mem_node = f'M:{mem}'

                # If a topic_id constraint is provided, optionally filter
                # memories either only at the final hop (default) or at every hop
                # when filter_each_hop is True.
                if topic_id is not None and topic_map is not None:
                    mem_topic = topic_map.get(str(mem))
                    should_filter = filter_each_hop or (next_hop == max_hops)
                    if should_filter and str(mem_topic) != str(topic_id):
                        # Skip adding this memory/edge entirely
                        continue

                # Append edge and mark reached
                edges.append((node, mem_node, next_hop))
                reached_any_memory = True

                if mem_node not in memory_hops:
                    memory_hops[mem_node] = next_hop
                    visited.add(mem_node)
                    queue.append((mem_node, next_hop))
            
            # If this term didn't reach any memory, it shouldn't be in the final graph
            if not reached_any_memory and node in term_hops:
                # This can happen if the term was added but leads nowhere
                pass
        
        elif node.startswith('M:'):
            # Memory → Terms (same hop, bridge to next memory)
            mem = node[2:]
            current_hop = last_mem_hop
            
            # Don't expand if we're at max_hops (no more memories to reach)
            if current_hop >= max_hops:
                continue
            
            for term in adj_by_memory.get(mem, []):
                if term in query_terms:
                    continue
                
                term_node = f'T:{term}'
                edges.append((node, term_node, current_hop))
                
                if term_node not in visited:
                    term_hops[term_node] = current_hop
                    visited.add(term_node)
                    queue.append((term_node, current_hop))
    
    # Clean up: remove terms that don't actually lead to memories
    # A term should only exist if it has outgoing edges to memories
    valid_terms = set()
    for src, tgt, hop in edges:
        if src.startswith('T:') and tgt.startswith('M:'):
            valid_terms.add(src)
    
    # Also keep query terms
    for qt in query_terms:
        valid_terms.add(f'T:{qt}')
    
    # Filter term_hops to only include valid terms
    term_hops = {t: h for t, h in term_hops.items() if t in valid_terms}
    
    # Filter edges to only include those connected to valid terms
    edges = [(src, tgt, hop) for src, tgt, hop in edges 
             if (not src.startswith('T:') or src in valid_terms) and
                (not tgt.startswith('T:') or tgt in valid_terms)]
    
    return memory_hops, term_hops, edges

def draw_multihop_graph(memory_hops, term_hops, edges, query_terms, max_hops, outdir,
                        topic_map=None, topic_id=None):
    """Draw multi-hop graph with spring layout.

    topic_map: dict mapping memory id -> topic value (strings)
    topic_id: if provided, memories matching this topic will be colored differently
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except Exception as e:
        print('Drawing requires networkx and matplotlib:', e)
        return
    
    G = nx.DiGraph()
    
    # Add all nodes
    for mem_node, hop in memory_hops.items():
        G.add_node(mem_node, hop=hop, node_type='memory')
    for term_node, hop in term_hops.items():
        G.add_node(term_node, hop=hop, node_type='term')
    
    # Add all edges
    for src, tgt, hop_level in edges:
        G.add_edge(src, tgt)
    
    # Spring layout
    print('Computing spring layout...')
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Draw
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=10, 
                          width=1.0, edge_color='gray', 
                          connectionstyle='arc3,rad=0.1', ax=ax)
    
    # Draw memory nodes by hop (different colors for different hops).
    # If a topic constraint is provided, color nodes that match the topic
    # differently from other topics.
    colors_mem = ['lightgreen', 'palegreen', 'lightcyan', 'lightyellow']
    # topic_map and topic_id are taken from the function args (passed from main)

    for hop in sorted(set(memory_hops.values())):
        mem_nodes_at_hop = [m for m, h in memory_hops.items() if h == hop]
        if not mem_nodes_at_hop:
            continue

        if topic_id is not None and topic_map is not None:
            same_topic = [m for m in mem_nodes_at_hop if topic_map.get(m[2:]) == topic_id]
            other_topic = [m for m in mem_nodes_at_hop if topic_map.get(m[2:]) != topic_id]

            if same_topic:
                nx.draw_networkx_nodes(G, pos, nodelist=same_topic,
                                      node_color='lightgreen', node_size=450,
                                      node_shape='o', edgecolors='darkgreen',
                                      linewidths=2, ax=ax, label=f'Memory Hop {hop} (Same Topic)')
            if other_topic:
                nx.draw_networkx_nodes(G, pos, nodelist=other_topic,
                                      node_color='lightgray', node_size=400,
                                      node_shape='o', edgecolors='darkslategray',
                                      linewidths=1.5, ax=ax, label=f'Memory Hop {hop} (Other Topic)')
        else:
            color = colors_mem[hop % len(colors_mem)]
            nx.draw_networkx_nodes(G, pos, nodelist=mem_nodes_at_hop, 
                                  node_color=color, node_size=400, 
                                  node_shape='o', edgecolors='darkgreen', 
                                  linewidths=2, ax=ax, label=f'Memory Hop {hop}')
    
    # Draw query terms (red squares)
    query_nodes = [f'T:{qt}' for qt in query_terms if f'T:{qt}' in term_hops]
    if query_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=query_nodes, 
                              node_color='red', node_size=600, 
                              node_shape='s', edgecolors='darkred', 
                              linewidths=3, ax=ax, label='Query Terms')
    
    # Draw other terms (blue circles, smaller)
    other_term_nodes = [t for t in term_hops.keys() if t not in query_nodes]
    if other_term_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=other_term_nodes, 
                              node_color='skyblue', node_size=250, 
                              node_shape='o', edgecolors='steelblue', 
                              linewidths=1.5, ax=ax, label='Intermediate Terms')
    
    # Labels for terms
    term_labels = {}
    for node in G.nodes():
        if node.startswith('T:'):
            label = node[2:]
            if len(label) > 20:
                label = label[:17] + '...'
            term_labels[node] = label
    
    nx.draw_networkx_labels(G, pos, term_labels, font_size=9, 
                           font_weight='bold', ax=ax)
    
    # Memory labels (smaller, green text)
    mem_labels = {}
    for node in memory_hops.keys():
        label = node[2:]
        if len(label) > 12:
            label = f"{label[:6]}..{label[-4:]}"
        mem_labels[node] = label
    
    nx.draw_networkx_labels(G, pos, mem_labels, font_size=7, 
                           font_color='darkgreen', ax=ax)
    
    # Title and legend
    n_mems = len(memory_hops)
    n_terms = len(term_hops)
    n_edges = len(edges)
    
    title = f'Multi-hop Graph (Spring Layout)\n'
    title += f'Query: {", ".join(query_terms[:3])}{"..." if len(query_terms) > 3 else ""}\n'
    title += f'Nodes: {n_mems} memories + {n_terms} terms | Edges: {n_edges} | Max Hops: {max_hops}'
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.axis('off')
    plt.tight_layout()
    
    png_path = outdir / 'multihop_graph.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f'✓ Saved visualization: {png_path}')
    print(f'  - {n_mems} memories, {n_terms} terms, {n_edges} edges')
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--entries', required=True)
    p.add_argument('--out-dir', default='./output')
    p.add_argument('--top-k', type=int, default=50)
    p.add_argument('--draw', action='store_true')
    p.add_argument('--draw-multihop', action='store_true')
    p.add_argument('--query-terms', type=str, default='')
    p.add_argument('--max-hops', type=int, default=3)
    p.add_argument('--topic-id', type=str, default=None,
                   help='Optional topic id to constrain memories (final hop by default)')
    p.add_argument('--filter-each-hop', action='store_true',
                   help='If set, apply the topic filter at every hop instead of only the final hop')
    args = p.parse_args()

    print('Building bipartite graph...')
    entries = load_entries(Path(args.entries))
    # Build a mapping from memory id -> topic value (supports several possible field names)
    # 修改后的代码
    topic_map = {}
    for rec in entries:
        mid = rec.get('id')
        if mid is None:
            continue
        
        # 首先尝试从顶层获取
        t = rec.get('topic_id')
        if t is None:
            t = rec.get('topic')
        if t is None:
            t = rec.get('topicId')
        
        # 如果顶层没有，尝试从 payload 中获取
        if t is None:
            payload = rec.get('payload', {})
            if isinstance(payload, dict):
                t = payload.get('topic_id')
                if t is None:
                    t = payload.get('topic')
                if t is None:
                    t = payload.get('topicId')
        
        if t is not None:
            # normalize keys/values to strings for stable comparisons
            topic_map[str(mid)] = str(t)
    adj_by_term, adj_by_memory = build_mappings(entries)
    
    # Basic outputs
    term_degrees = Counter({t: len(ms) for t, ms in adj_by_term.items()})
    top_terms = [t for t, _ in term_degrees.most_common(args.top_k)]
    mems = set()
    for t in top_terms:
        mems.update(adj_by_term[t])
    
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    write_json(list(top_terms), outdir / 'terms.json')
    write_json({t: sorted(list(adj_by_term[t])) for t in top_terms}, 
               outdir / 'adj_by_term.json')
    write_json({m: sorted(list(adj_by_memory[m])) for m in mems}, 
               outdir / 'adj_by_memory.json')
    
    edges = [[t, m] for t in top_terms for m in adj_by_term[t]]
    write_json(edges, outdir / 'edges.json')
    
    print(f'✓ Saved {len(top_terms)} terms, {len(mems)} memories, {len(edges)} edges')
    
    # Basic visualization
    if args.draw:
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except:
            print('Skipping basic draw (needs networkx, matplotlib)')
            return
        
        G = nx.Graph()
        for t in top_terms:
            G.add_node(f'T:{t}')
        for m in mems:
            G.add_node(f'M:{m}')
        for t, m in edges:
            G.add_edge(f'T:{t}', f'M:{m}')
        
        print('Computing spring layout for basic graph...')
        plt.figure(figsize=(14, 14))
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        term_nodes = [n for n in G.nodes() if n.startswith('T:')]
        mem_nodes = [n for n in G.nodes() if n.startswith('M:')]
        
        nx.draw_networkx_nodes(G, pos, nodelist=term_nodes, 
                              node_color='skyblue', node_size=300)
        nx.draw_networkx_nodes(G, pos, nodelist=mem_nodes, 
                              node_color='lightgreen', node_size=100)
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        nx.draw_networkx_labels(G, pos, {n: n[2:] for n in term_nodes}, font_size=7)
        
        plt.title(f'Bipartite Graph: Top {len(top_terms)} Terms', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(outdir / 'bipartite_basic.png', dpi=150)
        print('✓ Saved basic visualization')
        plt.close()
    
    # Multi-hop visualization
    if args.draw_multihop:
        if not args.query_terms:
            query_terms = top_terms[:3]
            print(f'Using top-3 terms as queries: {query_terms}')
        else:
            query_terms = [t.strip() for t in args.query_terms.split(',') if t.strip()]
        
        valid_queries = [qt for qt in query_terms if qt in adj_by_term]
        if not valid_queries:
            print('No valid query terms')
            return
        
        print(f'Running multi-hop BFS from: {valid_queries}')
        memory_hops, term_hops, edges_list = multi_hop_bfs(
            valid_queries, adj_by_term, adj_by_memory, args.max_hops,
            topic_map=topic_map, topic_id=args.topic_id,
            filter_each_hop=args.filter_each_hop
        )
        
        print(f'Found {len(memory_hops)} memories, {len(term_hops)} terms')
        # Print per-hop stats so that "Hop i" shows the terms that led
        # to the memories at hop i. Terms are at hop (i-1), memories at hop i.
        for hop in range(1, args.max_hops + 1):
            n_mems = sum(1 for h in memory_hops.values() if h == hop)
            n_terms = sum(1 for h in term_hops.values() if h == (hop - 1))
            if n_mems or n_terms:
                print(f'  Hop {hop}: {n_terms} terms, {n_mems} mems')
        
        draw_multihop_graph(memory_hops, term_hops, edges_list, 
                valid_queries, args.max_hops, outdir,
                topic_map=topic_map, topic_id=args.topic_id)

if __name__ == '__main__':
    main()