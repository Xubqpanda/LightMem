#!/usr/bin/env python3
"""
纯向量检索评测脚本 - LoCoMo数据集版本（支持按speaker或合并检索）

主要改进：
1. 默认按两个speaker合并排序检索top-60，可通过CLI切换回按speaker模式
2. 使用分speaker的prompt模板，明确区分两个说话人的记忆
3. 简化代码逻辑，去除冗余
4. 使用自定义生成的 summaries (从 hourly_aggregations.jsonl)
"""

from openai import OpenAI
import json
from tqdm import tqdm
import datetime
import time
import os
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import argparse

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
AGGREGATIONS_PATH = './hourly_with_summaries_v6.jsonl'

# ============ Retrieval Configuration ============
DEFAULT_RETRIEVAL_LIMIT = 60  # 默认检索数量


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


def load_custom_summaries(aggregations_path: str) -> Dict[str, List[Dict]]:
    """从 hourly_aggregations.jsonl 加载自定义生成的 summaries"""
    summaries_by_sample = {}
    
    with open(aggregations_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            record = json.loads(line)
            sample_id = record.get('collection', '')
            summary = record.get('summary', '')
            
            if not sample_id:
                continue
            
            if sample_id not in summaries_by_sample:
                summaries_by_sample[sample_id] = []
            
            summaries_by_sample[sample_id].append({
                'bucket': record.get('bucket', ''),
                'summary': summary,
                'vector': record.get('vector'),
                'speakers': record.get('speakers', []),
                'start_time': record.get('start_time', ''),
                'end_time': record.get('end_time', '')
            })
    
    script_logger.info(f"Loaded custom summaries for {len(summaries_by_sample)} samples")
    return summaries_by_sample




def build_session_summary_entries(sample_id: str, custom_summaries: Dict[str, List[Dict]], embedder) -> List[Dict[str, Any]]:
    """根据自定义 summaries 构建额外的检索条目"""
    sample_summaries = custom_summaries.get(sample_id, [])
    
    extra_entries: List[Dict[str, Any]] = []

    for summary_record in sample_summaries:
        summary_text = summary_record['summary']
        vector = summary_record.get('vector')
        bucket = summary_record['bucket']

        memory_text = f"[Session Summary] {summary_text}"

        payload = {
            'memory': memory_text,
            'memory_type': 'session_summary',
            'session_key': bucket,
            'source': 'session_summary',
            'speaker_name': 'Session Summary'
        }

        entry = {
            'id': f"{sample_id}__{bucket}",
            'payload': payload
        }

        if vector is not None:
            entry['vector'] = vector[:] if isinstance(vector, list) else vector

        extra_entries.append(entry)

    script_logger.info(
        f"[{sample_id}] Prepared {len(extra_entries)} session summary entries"
    )

    return extra_entries


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def retrieve_top_session_summaries(
    summary_entries: List[Dict[str, Any]],
    embedder,
    question: str,
    similarity_threshold: float = 0.5,
    max_results: int = 5
) -> List[Dict[str, Any]]:
    if not summary_entries:
        return []

    try:
        query_vector = embedder.embed(question)
    except Exception as e:
        script_logger.warning(f"Failed to embed question for session summary retrieval: {e}")
        return []

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for entry in summary_entries:
        vec = entry.get('vector')
        if vec is None:
            continue
        try:
            score = _cosine_similarity(query_vector, vec)
        except Exception as err:
            script_logger.warning(f"Failed to compute similarity for {entry.get('id')}: {err}")
            continue
        if score >= similarity_threshold:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        script_logger.info(
            f"No session summaries passed threshold {similarity_threshold:.2f}"
        )
        return []

    trimmed = scored[:max_results] if max_results > 0 else scored
    if len(scored) > len(trimmed):
        script_logger.info(
            f"Clipped session summaries from {len(scored)} to {len(trimmed)}"
        )

    return [entry for _, entry in trimmed]


def format_session_summaries(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return ""

    formatted: List[str] = []
    for entry in entries:
        payload = entry.get('payload', {}) if isinstance(entry, dict) else {}
        text = payload.get('memory') or entry.get('memory') or ""
        if text:
            formatted.append(text.strip())

    return "\n\n".join(formatted)

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

Session summaries:

{session_summaries}

Question: {question}

Answer:
"""



def retrieve_by_speaker(
    entries: List[Dict],
    retriever: VectorRetriever,
    question: str,
    limit_per_speaker: int = DEFAULT_RETRIEVAL_LIMIT
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


def retrieve_combined(
    entries: List[Dict],
    retriever: VectorRetriever,
    question: str,
    total_limit: int = DEFAULT_RETRIEVAL_LIMIT
) -> List[Dict]:
    """检索两个speaker合并后的top-k条记忆"""
    script_logger.info(f"Retrieving combined top-{total_limit} entries across speakers...")
    combined = retriever.retrieve(entries, question, limit=total_limit)
    for entry in combined:
        payload = entry.get('payload', {})
        entry['_retrieved_speaker'] = payload.get('speaker_name', 'Unknown')
    script_logger.info(f"  Combined retrieval returned {len(combined)} entries")
    return combined


def build_prompt_with_speaker_memories(
    question: str,
    retrieved_entries: List[Dict],
    session_summary_entries: Optional[List[Dict]] = None
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
        script_logger.info("speaker_1_memories:\n" + speaker_1_memories)
        script_logger.info("speaker_2_memories:\n" + speaker_2_memories)
    # Session summaries block
    if session_summary_entries:
        session_summaries_text = format_session_summaries(session_summary_entries)
    else:
        session_summaries_text = ""

    # 填充模板
    prompt = ANSWER_PROMPT.format(
        speaker_1_name=speaker_1_name,
        speaker_1_memories=speaker_1_memories,
        speaker_2_name=speaker_2_name,
        speaker_2_memories=speaker_2_memories,
        session_summaries=session_summaries_text,
        question=question
    )
    
    return prompt


def process_sample(
    sample: Dict,
    entry_loader: QdrantEntryLoader,
    retriever: VectorRetriever,
    llm: LLMModel,
    custom_summaries: Dict[str, List[Dict]],
    allow_categories: List[int],
    limit_per_speaker: int,
    total_limit: int,
    retrieval_mode: str,
    session_similarity_threshold: float
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

    summary_entries = build_session_summary_entries(sample_id, custom_summaries, retriever.embedder)
    if summary_entries:
        script_logger.info(f"[{sample_id}] Available session summaries: {len(summary_entries)}")
    
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
        
        # 选择检索策略
        time_start = time.time()
        if retrieval_mode == 'per-speaker':
            retrieved_entries = retrieve_by_speaker(entries, retriever, question, limit_per_speaker)
        else:
            retrieved_entries = retrieve_combined(entries, retriever, question, total_limit)
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
        
        # 构建prompt
        top_session_summaries = retrieve_top_session_summaries(
            summary_entries,
            retriever.embedder,
            question,
            similarity_threshold=session_similarity_threshold
        )

        user_prompt = build_prompt_with_speaker_memories(
            question,
            retrieved_entries,
            session_summary_entries=top_session_summaries
        )
        
        # 生成答案
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
    parser.add_argument('--aggregations-path', '--aggregations', type=str, dest='aggregations_path',
                       default=AGGREGATIONS_PATH, help="自定义 summaries 文件路径")
    parser.add_argument('--output-dir', type=str, default=RESULTS_DIR, help="输出目录")
    parser.add_argument('--limit-per-speaker', type=int, default=DEFAULT_RETRIEVAL_LIMIT,
                       help="每个speaker检索数量，当retrieval-mode=per-speaker时生效")
    parser.add_argument('--total-limit', type=int, default=DEFAULT_RETRIEVAL_LIMIT,
                       help="总检索数量，当retrieval-mode=combined时生效")
    parser.add_argument('--retrieval-mode', type=str, choices=['combined', 'per-speaker'],
                       default='combined', help="检索策略：combined为所有speaker合并排序取top-k")
    parser.add_argument('--allow-categories', type=int, nargs='+', default=[1, 2, 3, 4],
                       help="允许的category列表")
    parser.add_argument('--embedder', type=str, choices=['huggingface', 'openai'], 
                       default='huggingface', help='Embedding backend')
    parser.add_argument('--similarity-threshold', type=float, default=0.5,
                       help="会话总结向量过滤阈值")
    
    args = parser.parse_args()
    
    script_logger.info("=" * 80)
    script_logger.info("Vector Baseline Evaluation - LoCoMo Dataset (Speaker-based Retrieval)")
    script_logger.info(f"Config:")
    script_logger.info(f"  - Dataset: {args.dataset}")
    script_logger.info(f"  - Qdrant dir: {args.qdrant_dir}")
    script_logger.info(f"  - Aggregations: {args.aggregations_path}")
    script_logger.info(f"  - Output dir: {args.output_dir}")
    script_logger.info(f"  - Retrieval mode: {args.retrieval_mode}")
    if args.retrieval_mode == 'per-speaker':
        script_logger.info(f"  - Limit per speaker: {args.limit_per_speaker}")
    else:
        script_logger.info(f"  - Total limit: {args.total_limit}")
    script_logger.info(f"  - Allow categories: {args.allow_categories}")
    script_logger.info(f"  - Embedder: {args.embedder}")
    script_logger.info(f"  - Session similarity threshold: {args.similarity_threshold}")
    script_logger.info("=" * 80)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载自定义 summaries
    script_logger.info(f"\nLoading custom summaries from {args.aggregations_path}")
    custom_summaries = load_custom_summaries(args.aggregations_path)
    
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
            sample, entry_loader, retriever, llm, custom_summaries,
            args.allow_categories, args.limit_per_speaker,
            args.total_limit, args.retrieval_mode, args.similarity_threshold
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
        "aggregations_file": args.aggregations_path,
        "total_questions": total_questions,
        "total_samples": len(samples),
        "category_distribution": {str(cat): count for cat, count in category_counts.items()},
        "config": {
            "retrieval_mode": args.retrieval_mode,
            "limit_per_speaker": args.limit_per_speaker,
            "total_limit": args.total_limit,
            "embedding_model": EMBEDDING_MODEL_PATH,
            "method": "speaker_based_vector_retrieval_with_custom_summaries",
            "allow_categories": args.allow_categories,
            "session_similarity_threshold": args.similarity_threshold
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