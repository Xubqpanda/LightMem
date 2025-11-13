#!/usr/bin/env python3
"""
纯向量检索评测脚本 - LoCoMo数据集版本（按speaker分组检索）

主要改进：
1. 按speaker分别检索top-60（共60个），确保两个说话人的记忆平衡
2. 使用分speaker的prompt模板，明确区分两个说话人的记忆
3. 简化代码逻辑，去除冗余
"""

from openai import OpenAI
import json
from tqdm import tqdm
import datetime
import time
import os
import logging
from typing import List, Dict, Any
import numpy as np
import argparse
import pickle

# 导入 qdrant 相关
from lightmem.factory.text_embedder.huggingface import TextEmbedderHuggingface
from lightmem.factory.text_embedder.openai import TextEmbedderOpenAI
from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig

# 复用 retrievers 模块
from retrievers import LLMModel, QdrantEntryLoader, VectorRetriever, format_related_memories
from llm_judge import evaluate_llm_judge

# ============ Logging Configuration ============
LOGS_ROOT = "./logs"
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(LOGS_ROOT, f"vector_baseline_locomo_{RUN_TIMESTAMP}")
os.makedirs(RUN_LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(RUN_LOG_DIR, 'vector_baseline_locomo.log')),
        logging.StreamHandler()
    ]
)
script_logger = logging.getLogger("vector_baseline")

# ============ API Configuration ============
API_KEY = 'sk-elh8U89D8GUNipT9CdDaF39a1e5d44649e48B392E98cC24d'
API_BASE_URL = 'https://api.gpts.vin/v1'
LLM_MODEL = 'gpt-4o-mini'
JUDGE_MODEL = 'gpt-4o-mini'

# ============ Path Configuration ============
DATA_PATH = '/disk/disk_20T/xubuqiang/lightmem/dataset/locomo/locomo10.json'
QDRANT_DATA_DIR = './qdrant_data_locomo'
EMBEDDING_MODEL_PATH = '/disk/disk_20T/fangjizhan/models/all-MiniLM-L6-v2'
RESULTS_DIR = './results_vector_baseline_locomo'

# ============ Retrieval Configuration ============
RETRIEVAL_LIMIT_PER_SPEAKER = 60  # 每个speaker检索60个


def parse_locomo_dataset(data_path: str) -> List[Dict]:
    """解析LoCoMo数据集"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        sample = {
            'sample_id': item['sample_id'],
            'conversation': item['conversation'],
            'qa': []
        }
        
        for qa_item in item.get('qa', []):
            answer = qa_item.get('answer') or qa_item.get('adversarial_answer', '')
            sample['qa'].append({
                'question': qa_item['question'],
                'answer': answer,
                'category': qa_item['category']
            })
        
        samples.append(sample)
    
    return samples

# ============ Prompt Template ============
ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain 
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.), 
   calculate the actual date based on the memory timestamp. For example, if a memory from 
   4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example, 
   convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
   timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories from both speakers. Do not confuse character 
   names mentioned in memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Memories for user {speaker_1_name}:

{speaker_1_memories}

Memories for user {speaker_2_name}:

{speaker_2_memories}

Question: {question}

Answer:
"""



def retrieve_by_speaker(
    entries: List[Dict],
    retriever: VectorRetriever,
    question: str,
    limit_per_speaker: int = 60
) -> List[Dict]:
    """
    按speaker分组检索，对每个speaker分别检索top-k个记忆
    
    Returns:
        合并后的检索结果列表，每个entry标记了 _retrieved_speaker
    """
    # 按speaker_name分组
    speaker_groups = {}
    for entry in entries:
        payload = entry.get('payload', {})
        speaker_name = payload.get('speaker_name', 'Unknown')
        
        if speaker_name not in speaker_groups:
            speaker_groups[speaker_name] = []
        speaker_groups[speaker_name].append(entry)
    
    script_logger.info(f"Found {len(speaker_groups)} speakers: {list(speaker_groups.keys())}")
    
    # 对每个speaker分别检索
    all_retrieved = []
    for speaker_name, group_entries in speaker_groups.items():
        script_logger.info(f"Retrieving top-{limit_per_speaker} for {speaker_name}...")
        
        speaker_retrieved = retriever.retrieve(
            group_entries, 
            question, 
            limit=limit_per_speaker
        )
        
        script_logger.info(f"  Retrieved {len(speaker_retrieved)}/{len(group_entries)} entries")
        
        # 标记speaker
        for entry in speaker_retrieved:
            entry['_retrieved_speaker'] = speaker_name
        
        all_retrieved.extend(speaker_retrieved)
    
    return all_retrieved


def build_prompt_with_speaker_memories(
    question: str,
    retrieved_entries: List[Dict],
    sample_id: str,
    retriever: VectorRetriever,
    enable_graph: bool = False,
    graph_dir: str = './temporal_events_json_v3',
    graph_threshold: float = 0.7,
) -> str:
    """
    构建包含分speaker记忆的prompt
    
    复用 format_related_memories 来格式化每个speaker的记忆
    """
    # 按speaker分组
    speaker_groups = {}
    for entry in retrieved_entries:
        speaker_name = entry.get('_retrieved_speaker', 
                                 entry.get('payload', {}).get('speaker_name', 'Unknown'))
        if speaker_name not in speaker_groups:
            speaker_groups[speaker_name] = []
        speaker_groups[speaker_name].append(entry)
    
    speaker_names = list(speaker_groups.keys())
    
    # 处理不同情况
    if len(speaker_names) == 0:
        speaker_1_name = "Speaker 1"
        speaker_2_name = "Speaker 2"
        speaker_1_memories = "No memories available."
        speaker_2_memories = "No memories available."
    elif len(speaker_names) == 1:
        speaker_1_name = speaker_names[0]
        speaker_2_name = "Speaker 2"
        speaker_1_memories = format_related_memories(speaker_groups[speaker_1_name])
        speaker_2_memories = "No memories available."
    else:
        speaker_1_name = speaker_names[0]
        speaker_2_name = speaker_names[1]
        speaker_1_memories = format_related_memories(speaker_groups[speaker_1_name])
        speaker_2_memories = format_related_memories(speaker_groups[speaker_2_name])
        
        script_logger.info(
            f"Formatted memories - {speaker_1_name}: {len(speaker_groups[speaker_1_name])}, "
            f"{speaker_2_name}: {len(speaker_groups[speaker_2_name])}"
        )
        script_logger.info(f"speaker_1_memories: {speaker_1_memories}")
        script_logger.info(f"speaker_2_memories: {speaker_2_memories}")
    # 如果启用了 graph，把匹配到的 graph events 拼接进对应 speaker 的 memories
    if enable_graph:
        try:
            graph_file = os.path.join(graph_dir, f"{sample_id}_events.json")
            if os.path.exists(graph_file):
                with open(graph_file, 'r', encoding='utf-8') as gf:
                    graph_data = json.load(gf)
                events = graph_data.get('data', []) if isinstance(graph_data, dict) else []

                # 预先计算 question embedding
                try:
                    q_emb = retriever.embedder.embed(question)
                except Exception as e:
                    script_logger.warning(f"Failed to embed question for graph matching: {e}")
                    q_emb = None

                # 尝试从缓存加载 event embeddings（按 sample 缓存）
                cache_file = os.path.join(graph_dir, f"{sample_id}_events_embeddings.pkl")
                event_embeddings = None
                try:
                    if os.path.exists(cache_file):
                        with open(cache_file, 'rb') as cf:
                            obj = pickle.load(cf)
                        # 期望 obj 为 dict {'descs': [...], 'embeddings': [...]} 与 events 顺序一致
                        if isinstance(obj, dict) and 'embeddings' in obj:
                            event_embeddings = obj['embeddings']
                except Exception as e:
                    script_logger.warning(f"Failed to load event embedding cache {cache_file}: {e}")

                def cos_sim(a, b):
                    if a is None or b is None:
                        return 0.0
                    a = np.array(a)
                    b = np.array(b)
                    na = np.linalg.norm(a)
                    nb = np.linalg.norm(b)
                    if na == 0 or nb == 0:
                        return 0.0
                    return float(np.dot(a, b) / (na * nb))

                # collect matches per speaker
                graph_matches = {}
                # 如果没有缓存或缓存不匹配，则计算所有 event embeddings 并写入缓存
                if event_embeddings is None or len(event_embeddings) != len(events):
                    calc_embeddings = []
                    for ev in events:
                        desc = ev.get('Event_description', '')
                        if not desc:
                            calc_embeddings.append(None)
                            continue
                        try:
                            e_emb = retriever.embedder.embed(desc)
                        except Exception as e:
                            script_logger.warning(f"Failed to embed event desc for {sample_id}: {e}")
                            e_emb = None
                        calc_embeddings.append(e_emb)
                    event_embeddings = calc_embeddings
                    try:
                        with open(cache_file, 'wb') as cf:
                            pickle.dump({'descs': [ev.get('Event_description', '') for ev in events], 'embeddings': event_embeddings}, cf)
                    except Exception as e:
                        script_logger.warning(f"Failed to write event embedding cache {cache_file}: {e}")

                for ev_idx, ev in enumerate(events):
                    desc = ev.get('Event_description', '')
                    speaker = ev.get('Speaker', '')
                    if not desc or not speaker:
                        continue
                    e_emb = None
                    if event_embeddings and ev_idx < len(event_embeddings):
                        e_emb = event_embeddings[ev_idx]

                    sim = cos_sim(q_emb, e_emb)
                    script_logger.info(f"Graph sim={sim:.4f} for speaker {speaker}: {desc}")
                    if sim >= float(graph_threshold):
                        timeline = ev.get('Event_timeline', [])
                        if isinstance(timeline, list):
                            timeline_str = "\n".join(timeline)
                        else:
                            timeline_str = str(timeline)

                        formatted = f"[GRAPH_EVENT] {desc}\nTimeline:\n{timeline_str}"
                        script_logger.info(f"Graph match (sim={sim:.4f}) for speaker {speaker}: {formatted}")
                        graph_matches.setdefault(speaker, []).append(formatted)

                # 拼接到对应 speaker memories
                for spk, items in graph_matches.items():
                    add_text = "\n\n" + "\n\n".join(items)
                    # 如果该 speaker 在当前检索到的 groups里，附加到对应 memory string
                    if spk == speaker_1_name:
                        speaker_1_memories = speaker_1_memories + add_text if speaker_1_memories else add_text
                    elif spk == speaker_2_name:
                        speaker_2_memories = speaker_2_memories + add_text if speaker_2_memories else add_text
                    else:
                        # 如果 speaker 名称不在前两个，尝试附加到第一个speaker
                        speaker_1_memories = speaker_1_memories + add_text if speaker_1_memories else add_text

        except Exception as e:
            script_logger.warning(f"Graph matching failed for {sample_id}: {e}")


    # 填充模板
    prompt = ANSWER_PROMPT.format(
        speaker_1_name=speaker_1_name,
        speaker_1_memories=speaker_1_memories,
        speaker_2_name=speaker_2_name,
        speaker_2_memories=speaker_2_memories,
        question=question
    )

    return prompt


def process_sample(
    sample: Dict,
    entry_loader: QdrantEntryLoader,
    retriever: VectorRetriever,
    llm: LLMModel,
    allow_categories: List[int],
    limit_per_speaker: int,
    enable_graph: bool = False,
    graph_dir: str = './temporal_events_json_v3',
    graph_threshold: float = 0.7,
) -> Dict:
    """处理单个sample的所有QA"""
    sample_id = sample['sample_id']
    script_logger.info(f"\n{'='*80}")
    script_logger.info(f"Processing sample: {sample_id}")
    script_logger.info(f"{'='*80}")
    
    # 加载entries
    try:
        entries = entry_loader.load_entries(sample_id, with_vectors=True)
        if not entries:
            script_logger.error(f"[{sample_id}] No entries loaded")
            return {'sample_id': sample_id, 'error': 'No entries loaded', 'results': []}
        
        script_logger.info(f"[{sample_id}] Loaded {len(entries)} entries")
    except Exception as e:
        script_logger.error(f"[{sample_id}] Failed to load entries: {e}")
        return {'sample_id': sample_id, 'error': str(e), 'results': []}
    
    # 处理每个QA
    qa_results = []
    for qa_idx, qa in enumerate(sample['qa']):
        category = qa['category']
        
        # 跳过category 5和不允许的category
        if int(category) == 5 or category not in allow_categories:
            continue
        
        question = qa['question']
        reference = qa['answer']
        
        script_logger.info(f"\n[{sample_id}] Question {qa_idx+1} (Category {category})")
        script_logger.info(f"Q: {question}")
        script_logger.info(f"A: {reference}")
        
        # 按speaker检索
        time_start = time.time()
        retrieved_entries = retrieve_by_speaker(entries, retriever, question, limit_per_speaker)
        retrieval_time = time.time() - time_start
        
        if not retrieved_entries:
            script_logger.warning(f"[{sample_id}] No entries retrieved")
            qa_results.append({
                'question': question,
                'prediction': '',
                'reference': reference,
                'category': category,
                'retrieved_count': 0,
                'retrieval_time': retrieval_time,
                'speaker_distribution': {},
                'error': 'No entries retrieved',
                'metrics': {}
            })
            continue
        
        # 统计speaker分布
        speaker_dist = {}
        for entry in retrieved_entries:
            speaker = entry.get('_retrieved_speaker', 'Unknown')
            speaker_dist[speaker] = speaker_dist.get(speaker, 0) + 1
        
        script_logger.info(
            f"[{sample_id}] Retrieved {len(retrieved_entries)} entries in {retrieval_time:.3f}s"
        )
        script_logger.info(f"[{sample_id}] Speaker distribution: {speaker_dist}")
        
        user_prompt = build_prompt_with_speaker_memories(
            question, retrieved_entries, sample_id, retriever,
            enable_graph=enable_graph, graph_dir=graph_dir,
            graph_threshold=graph_threshold
        )
        
        try:
            openai_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

            response = openai_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {"role": "system", "content": user_prompt}
                ],
                temperature=0.0
            )
            generated_answer = response.choices[0].message.content
            script_logger.info(f"[{sample_id}] Generated: {generated_answer}")
        except Exception as e:
            script_logger.error(f"[{sample_id}] Failed to generate answer: {e}")
            generated_answer = ""
        
        # LLM judge评估
        try:
            label = evaluate_llm_judge(
                question, reference, generated_answer, 
                client_obj=llm.client, model_name=JUDGE_MODEL
            )
            metrics = {
                'judge_correct': int(label),
                'judge_response': 'CORRECT' if int(label) == 1 else 'WRONG'
            }
            script_logger.info(
                f"[{sample_id}] Judge: {'CORRECT' if int(label) == 1 else 'WRONG'}"
            )
        except Exception as e:
            script_logger.error(f"[{sample_id}] Judge failed: {e}")
            metrics = {'judge_correct': 0, 'judge_response': ''}
        
        # 保存结果
        qa_results.append({
            'question': question,
            'prediction': generated_answer,
            'reference': reference,
            'category': category,
            'retrieved_count': len(retrieved_entries),
            'speaker_distribution': speaker_dist,
            'retrieval_time': retrieval_time,
            'metrics': metrics
        })
    
    return {'sample_id': sample_id, 'results': qa_results}


def main():
    parser = argparse.ArgumentParser(description="纯向量检索评测 - LoCoMo数据集（按speaker分组）")
    parser.add_argument('--dataset', type=str, default=DATA_PATH, help="数据集路径")
    parser.add_argument('--qdrant-dir', type=str, default=QDRANT_DATA_DIR, help="Qdrant数据目录")
    parser.add_argument('--output-dir', type=str, default=RESULTS_DIR, help="输出目录")
    parser.add_argument('--limit-per-speaker', type=int, default=RETRIEVAL_LIMIT_PER_SPEAKER,
                       help="每个speaker检索数量")
    parser.add_argument('--allow-categories', type=int, nargs='+', default=[1, 2, 3, 4],
                       help="允许的category列表")
    parser.add_argument('--embedder', type=str, choices=['huggingface', 'openai'], 
                       default='huggingface', help='Embedding backend')
    parser.add_argument('--enable-graph', action='store_true', help='是否将graph memory加入检索')
    parser.add_argument('--graph-dir', type=str, default='./temporal_events_json_v3', help='Graph JSON 目录（每个 sample 对应 sample_id_events.json）')
    parser.add_argument('--graph-threshold', type=float, default=0.7, help='Graph matching similarity threshold (cosine)')
    
    args = parser.parse_args()
    
    script_logger.info("=" * 80)
    script_logger.info("Vector Baseline Evaluation - LoCoMo Dataset (Speaker-based Retrieval)")
    script_logger.info(f"Config:")
    script_logger.info(f"  - Dataset: {args.dataset}")
    script_logger.info(f"  - Qdrant dir: {args.qdrant_dir}")
    script_logger.info(f"  - Output dir: {args.output_dir}")
    script_logger.info(f"  - Limit per speaker: {args.limit_per_speaker}")
    script_logger.info(f"  - Allow categories: {args.allow_categories}")
    script_logger.info(f"  - Embedder: {args.embedder}")
    script_logger.info("=" * 80)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化组件
    script_logger.info("\nInitializing components...")
    entry_loader = QdrantEntryLoader(args.qdrant_dir)
    
    # 初始化embedding模型
    if args.embedder == 'openai':
        embedder_cfg = BaseTextEmbedderConfig(
            model='text-embedding-3-small',
            api_key=API_KEY,
            openai_base_url=API_BASE_URL,
            embedding_dims=1536,
        )
        embedder = TextEmbedderOpenAI(embedder_cfg)
    else:
        embedder_cfg = BaseTextEmbedderConfig(
            model=EMBEDDING_MODEL_PATH,
            embedding_dims=384,
            model_kwargs={"device": "cuda"},
        )
        embedder = TextEmbedderHuggingface(embedder_cfg)
    
    retriever = VectorRetriever(embedder)
    llm = LLMModel(LLM_MODEL, API_KEY, API_BASE_URL)
    
    # 加载数据集
    script_logger.info(f"\nLoading dataset from {args.dataset}")
    samples = parse_locomo_dataset(args.dataset)
    script_logger.info(f"Loaded {len(samples)} samples")
    
    # 处理所有样本
    all_results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    
    for sample in tqdm(samples, desc="Processing samples"):
        sample_result = process_sample(
            sample, entry_loader, retriever, llm, 
            args.allow_categories, args.limit_per_speaker,
            enable_graph=args.enable_graph, graph_dir=args.graph_dir,
            graph_threshold=args.graph_threshold
        )
        
        all_results.append(sample_result)
        
        # 汇总指标
        for qa_result in sample_result.get('results', []):
            total_questions += 1
            category = qa_result['category']
            category_counts[category] += 1
            all_metrics.append(qa_result['metrics'])
            all_categories.append(category)
        
        # 保存单个sample结果
        sample_file = os.path.join(args.output_dir, f"sample_{sample['sample_id']}.json")
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_result, f, ensure_ascii=False, indent=2)
    
    # 计算聚合指标
    script_logger.info("\nCalculating aggregate metrics...")
    category_scores = {}
    total_scores = []
    
    for cat, m in zip(all_categories, all_metrics):
        score = float(m.get('judge_correct', 0)) if isinstance(m, dict) else 0.0
        total_scores.append(score)
        category_scores.setdefault(int(cat), []).append(score)
    
    aggregate_results = {"overall": {}}
    if total_scores:
        aggregate_results["overall"]["judge_correct"] = {
            "mean": float(np.mean(total_scores)),
            "std": float(np.std(total_scores)),
            "count": int(len(total_scores)),
        }
    
    for cat in sorted(category_scores.keys()):
        vals = category_scores[cat]
        if vals:
            aggregate_results[f"category_{cat}"] = {
                "judge_correct": {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "count": int(len(vals)),
                }
            }
    
    # 保存最终结果
    final_results = {
        "model": LLM_MODEL,
        "dataset": args.dataset,
        "total_questions": total_questions,
        "total_samples": len(samples),
        "category_distribution": {str(cat): count for cat, count in category_counts.items()},
        "config": {
            "limit_per_speaker": args.limit_per_speaker,
            "embedding_model": EMBEDDING_MODEL_PATH,
            "method": "speaker_based_vector_retrieval",
            "allow_categories": args.allow_categories
        },
        "aggregate_metrics": aggregate_results,
        "timestamp": RUN_TIMESTAMP
    }
    
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    # 打印汇总信息
    script_logger.info("\n" + "=" * 80)
    script_logger.info("Evaluation completed!")
    script_logger.info(f"Total samples: {len(samples)}")
    script_logger.info(f"Total questions: {total_questions}")
    script_logger.info("\nCategory Distribution:")
    for category, count in sorted(category_counts.items()):
        if count > 0:
            script_logger.info(
                f"  Category {category}: {count} questions "
                f"({count/total_questions*100:.1f}%)"
            )
    
    script_logger.info("\nAggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        script_logger.info(f"\n{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            if isinstance(stats, dict):
                script_logger.info(f"  {metric_name}:")
                for stat_name, value in stats.items():
                    script_logger.info(f"    {stat_name}: {value:.4f}")
    
    script_logger.info(f"\nResults saved to: {args.output_dir}")
    script_logger.info("=" * 80)


if __name__ == "__main__":
    main()