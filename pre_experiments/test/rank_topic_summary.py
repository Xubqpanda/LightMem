"""
简化版 Topic Summary 检索实验

核心逻辑：
1. 按topic_id去重，提取唯一的topic_summary
2. 对topic_summary进行embedding
3. 计算query与topic_summary的相似度
4. 查看相关topic的排名
"""

import json
import argparse
import numpy as np
from typing import Dict, List, Tuple


def load_embedder(model_path: str = "/disk/disk_20T/fangjizhan/models/all-MiniLM-L6-v2"):
    """加载embedding模型"""
    try:
        from sentence_transformers import SentenceTransformer
        print(f"加载embedding模型: {model_path}")
        model = SentenceTransformer(model_path)
        return model
    except ImportError:
        raise RuntimeError("需要安装sentence-transformers: pip install sentence-transformers")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


def extract_unique_topics(json_path: str) -> Dict[str, str]:
    """
    从JSON文件中提取唯一的topic_id和对应的topic_summary
    
    Returns:
        {topic_id: topic_summary}
    """
    print(f"加载数据: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    
    print(f"共加载 {len(entries)} 条记录")
    
    # 按topic_id去重
    topic_dict = {}
    no_topic_count = 0
    
    for entry in entries:
        payload = entry.get('payload', {})
        topic_id = payload.get('topic_id')
        topic_summary = payload.get('topic_summary')
        
        # 跳过没有topic_id或topic_summary的记录
        if topic_id is None or not topic_summary:
            no_topic_count += 1
            continue
        
        topic_id = str(topic_id)
        
        # 如果这个topic_id第一次出现，记录它的summary
        if topic_id not in topic_dict:
            topic_dict[topic_id] = topic_summary
    
    print(f"去重后共有 {len(topic_dict)} 个唯一的topic")
    print(f"跳过了 {no_topic_count} 条没有topic信息的记录")
    
    return topic_dict


def count_memories_per_topic(json_path: str) -> Dict[str, int]:
    """
    统计每个topic包含多少条memory
    
    Returns:
        {topic_id: memory_count}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    
    topic_counts = {}
    
    for entry in entries:
        payload = entry.get('payload', {})
        topic_id = payload.get('topic_id')
        
        if topic_id is not None:
            topic_id = str(topic_id)
            topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
    
    return topic_counts


def embed_topics(topic_dict: Dict[str, str], embedder) -> Dict[str, np.ndarray]:
    """
    对所有topic_summary进行embedding
    
    Args:
        topic_dict: {topic_id: topic_summary}
        embedder: embedding模型
    
    Returns:
        {topic_id: embedding_vector}
    """
    print("\n正在embedding所有topic_summary...")
    
    topic_ids = list(topic_dict.keys())
    topic_summaries = [topic_dict[tid] for tid in topic_ids]
    
    # 批量embedding
    print(f"共 {len(topic_summaries)} 个topic需要embedding")
    embeddings = embedder.encode(topic_summaries, show_progress_bar=True)
    
    # 构建结果字典
    topic_embeddings = {}
    for topic_id, embedding in zip(topic_ids, embeddings):
        topic_embeddings[topic_id] = embedding
    
    print(f"完成！")
    return topic_embeddings


def retrieve_topics(
    query: str,
    topic_dict: Dict[str, str],
    topic_embeddings: Dict[str, np.ndarray],
    topic_counts: Dict[str, int],
    embedder,
    top_k: int = 20
) -> List[Tuple[str, float, str, int]]:
    """
    使用query检索最相关的topics
    
    Returns:
        List of (topic_id, similarity_score, topic_summary, memory_count)
    """
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}")
    
    # Embed query
    print("正在embedding query...")
    query_embedding = embedder.encode(query)
    
    # 计算相似度
    print("计算相似度...")
    results = []
    for topic_id, topic_embedding in topic_embeddings.items():
        similarity = cosine_similarity(query_embedding, topic_embedding)
        topic_summary = topic_dict[topic_id]
        memory_count = topic_counts.get(topic_id, 0)
        results.append((topic_id, similarity, topic_summary, memory_count))
    
    # 按相似度排序
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:top_k]


def print_results(results: List[Tuple[str, float, str, int]]):
    """打印检索结果"""
    print(f"\nTop {len(results)} 相关Topics:")
    print(f"{'='*80}")
    
    for rank, (topic_id, score, topic_summary, memory_count) in enumerate(results, 1):
        print(f"\n[Rank {rank}] Topic ID: {topic_id}")
        print(f"相似度得分: {score:.4f}")
        print(f"Topic Summary: {topic_summary}")
        print(f"包含 {memory_count} 条memories")
        print(f"{'-'*80}")


def find_target_topic_rank(
    target_topic_id: str,
    results: List[Tuple[str, float, str, int]]
) -> Tuple[int, float]:
    """
    在结果中找到目标topic的排名
    
    Returns:
        (rank, score) 如果没找到返回 (-1, 0.0)
    """
    for rank, (topic_id, score, _, _) in enumerate(results, 1):
        if topic_id == target_topic_id:
            return rank, score
    return -1, 0.0


def analyze_all_topics(
    query: str,
    topic_dict: Dict[str, str],
    topic_embeddings: Dict[str, np.ndarray],
    embedder
) -> List[Tuple[str, float, str]]:
    """
    计算query与所有topic的相似度（不做top-k截断）
    用于详细分析
    
    Returns:
        List of (topic_id, similarity_score, topic_summary) 按相似度排序
    """
    query_embedding = embedder.encode(query)
    
    results = []
    for topic_id, topic_embedding in topic_embeddings.items():
        similarity = cosine_similarity(query_embedding, topic_embedding)
        topic_summary = topic_dict[topic_id]
        results.append((topic_id, similarity, topic_summary))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="简化版Topic Summary检索实验")
    parser.add_argument(
        "--json",
        required=True,
        help="导出的JSON文件路径"
    )
    parser.add_argument(
        "--query",
        required=True,
        help="测试查询语句"
    )
    parser.add_argument(
        "--model",
        default="/disk/disk_20T/fangjizhan/models/all-MiniLM-L6-v2",
        help="Embedding模型路径（默认: all-MiniLM-L6-v2）"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="显示top-k结果（默认: 20）"
    )
    parser.add_argument(
        "--target-topic",
        type=str,
        default=None,
        help="指定目标topic_id，查看它在所有结果中的排名"
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="显示所有topic的排名（不限制top-k）"
    )
    
    args = parser.parse_args()
    
    # 1. 加载数据并去重
    topic_dict = extract_unique_topics(args.json)
    
    if len(topic_dict) == 0:
        print("错误：没有找到任何包含topic_summary的数据")
        return
    
    # 2. 统计每个topic的memory数量
    topic_counts = count_memories_per_topic(args.json)
    
    # 3. 加载embedding模型
    embedder = load_embedder(args.model)
    
    # 4. Embedding所有topic_summary
    topic_embeddings = embed_topics(topic_dict, embedder)
    
    # 5. 执行检索
    if args.show_all:
        # 显示所有topic的排名
        all_results = analyze_all_topics(args.query, topic_dict, topic_embeddings, embedder)
        print(f"\n{'='*80}")
        print(f"Query: {args.query}")
        print(f"{'='*80}")
        print(f"\n所有 {len(all_results)} 个Topics的排名:")
        print(f"{'='*80}")
        
        for rank, (topic_id, score, topic_summary) in enumerate(all_results, 1):
            memory_count = topic_counts.get(topic_id, 0)
            print(f"[{rank}] Topic {topic_id} | Score: {score:.4f} | Memories: {memory_count}")
            print(f"    {topic_summary}")
            print(f"    {'-'*76}")
    else:
        # 只显示top-k
        results = retrieve_topics(
            args.query,
            topic_dict,
            topic_embeddings,
            topic_counts,
            embedder,
            args.top_k
        )
        print_results(results)
    
    # 6. 如果指定了目标topic，特别标注
    if args.target_topic:
        print(f"\n{'='*80}")
        print(f"目标Topic分析:")
        print(f"{'='*80}")
        
        # 计算所有topic的相似度来找排名
        all_results = analyze_all_topics(args.query, topic_dict, topic_embeddings, embedder)
        
        rank = -1
        score = 0.0
        for r, (tid, s, _) in enumerate(all_results, 1):
            if tid == args.target_topic:
                rank = r
                score = s
                break
        
        if rank > 0:
            print(f"Topic ID: {args.target_topic}")
            print(f"在所有 {len(all_results)} 个topics中排名: 第 {rank} 位")
            print(f"相似度得分: {score:.4f}")
            print(f"Topic Summary: {topic_dict.get(args.target_topic, 'N/A')}")
            print(f"包含 {topic_counts.get(args.target_topic, 0)} 条memories")
            
            if rank <= args.top_k:
                print(f"\n✓ 该topic在top-{args.top_k}中!")
            else:
                print(f"\n✗ 该topic未进入top-{args.top_k} (差 {rank - args.top_k} 位)")
        else:
            print(f"错误：Topic ID '{args.target_topic}' 不存在")


if __name__ == '__main__':
    main()