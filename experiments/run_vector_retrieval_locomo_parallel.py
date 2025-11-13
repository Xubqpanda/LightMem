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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import hashlib
from lightmem.factory.text_embedder.huggingface import TextEmbedderHuggingface
from lightmem.factory.text_embedder.openai import TextEmbedderOpenAI
from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig
from retrievers import LLMModel, QdrantEntryLoader, VectorRetriever, format_related_memories
from llm_judge import evaluate_llm_judge

# ============ Logging Configuration ============
LOGS_ROOT = "./logs"
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(LOGS_ROOT, f"vector_baseline_locomo_parallel_{RUN_TIMESTAMP}")
os.makedirs(RUN_LOG_DIR, exist_ok=True)

# ============ API Configuration ============
API_KEY = 'sk-elh8U89D8GUNipT9CdDaF39a1e5d44649e48B392E98cC24d'
API_BASE_URL = 'https://api.gpts.vin/v1'
LLM_MODEL = 'gpt-4o-mini'
JUDGE_MODEL = 'gpt-4o-mini'

# ============ Path Configuration ============
DATA_PATH = '/disk/disk_20T/xubuqiang/lightmem/dataset/locomo/locomo10.json'
QDRANT_DATA_DIR = './qdrant_data_locomo'
EMBEDDING_MODEL_PATH = '/disk/disk_20T/fangjizhan/models/all-MiniLM-L6-v2'
RESULTS_DIR = './results_vector_baseline_locomo_parallel'

# ============ GPU Configuration ============
AVAILABLE_GPUS = [0, 1, 2, 3] 
MAX_WORKERS = 10 

# ============ Retrieval Configuration ============
RETRIEVAL_LIMIT_PER_SPEAKER = 60


def get_gpu_for_sample(sample_id: str, available_gpus: List[int]) -> int:
    hash_value = int(hashlib.md5(sample_id.encode()).hexdigest(), 16)
    gpu_id = available_gpus[hash_value % len(available_gpus)]
    return gpu_id


def get_process_logger(sample_id: str) -> logging.Logger:
    logger = logging.getLogger(f"vector_eval.{sample_id}")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        fh = logging.FileHandler(
            os.path.join(RUN_LOG_DIR, f"{sample_id}.log"),
            mode='w'
        )
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


def parse_locomo_dataset(data_path: str) -> List[Dict]:
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
    limit_per_speaker: int,
    logger: logging.Logger
) -> List[Dict]:
    speaker_groups = {}
    for entry in entries:
        payload = entry.get('payload', {})
        speaker_name = payload.get('speaker_name', 'Unknown')
        
        if speaker_name not in speaker_groups:
            speaker_groups[speaker_name] = []
        speaker_groups[speaker_name].append(entry)
    
    logger.debug(f"Found {len(speaker_groups)} speakers: {list(speaker_groups.keys())}")
    
    all_retrieved = []
    for speaker_name, group_entries in speaker_groups.items():
        logger.debug(f"Retrieving top-{limit_per_speaker} for {speaker_name}...")
        
        speaker_retrieved = retriever.retrieve(
            group_entries, 
            question, 
            limit=limit_per_speaker
        )
        
        logger.debug(f"  Retrieved {len(speaker_retrieved)}/{len(group_entries)} entries")
        
        for entry in speaker_retrieved:
            entry['_retrieved_speaker'] = speaker_name
        
        all_retrieved.extend(speaker_retrieved)
    
    return all_retrieved


def build_prompt_with_speaker_memories(question: str, retrieved_entries: List[Dict]) -> str:
    speaker_groups = {}
    for entry in retrieved_entries:
        speaker_name = entry.get('_retrieved_speaker', 
                                 entry.get('payload', {}).get('speaker_name', 'Unknown'))
        if speaker_name not in speaker_groups:
            speaker_groups[speaker_name] = []
        speaker_groups[speaker_name].append(entry)
    
    speaker_names = list(speaker_groups.keys())
    
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
    
    prompt = ANSWER_PROMPT.format(
        speaker_1_name=speaker_1_name,
        speaker_1_memories=speaker_1_memories,
        speaker_2_name=speaker_2_name,
        speaker_2_memories=speaker_2_memories,
        question=question
    )
    
    return prompt


def process_single_sample_with_gpu(
    sample: Dict,
    gpu_id: int,
    embedder_type: str,
    qdrant_dir: str,
    allow_categories: List[int],
    limit_per_speaker: int,
    output_dir: str
) -> Dict:
    sample_id = sample['sample_id']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    logger = get_process_logger(sample_id)
    logger.info(f"✓ Assigned to GPU {gpu_id}")
    
    try:
        logger.info(f"{'='*80}")
        logger.info(f"Processing sample: {sample_id} on GPU {gpu_id}")
        logger.info(f"{'='*80}")
        
        entry_loader = QdrantEntryLoader(qdrant_dir)
        
        if embedder_type == 'openai':
            embedder_cfg = BaseTextEmbedderConfig(
                model='text-embedding-3-small',
                api_key=API_KEY,
                openai_base_url=API_BASE_URL,
                embedding_dims=1536,
            )
            embedder = TextEmbedderOpenAI(embedder_cfg)
            logger.info("Initialized OpenAI embedder")
        else:
            embedder_cfg = BaseTextEmbedderConfig(
                model=EMBEDDING_MODEL_PATH,
                embedding_dims=384,
                model_kwargs={"device": "cuda"},  
            )
            embedder = TextEmbedderHuggingface(embedder_cfg)
            logger.info(f"Initialized HuggingFace embedder on GPU {gpu_id}")
        
        retriever = VectorRetriever(embedder)
        llm = LLMModel(LLM_MODEL, API_KEY, API_BASE_URL)
        
        try:
            entries = entry_loader.load_entries(sample_id, with_vectors=True)
            if not entries:
                logger.error(f"No entries loaded")
                return {
                    'sample_id': sample_id, 
                    'gpu_id': gpu_id,
                    'error': 'No entries loaded', 
                    'results': []
                }
            
            logger.info(f"Loaded {len(entries)} entries")
        except Exception as e:
            logger.error(f"Failed to load entries: {e}", exc_info=True)
            return {
                'sample_id': sample_id, 
                'gpu_id': gpu_id,
                'error': str(e), 
                'results': []
            }
        
        qa_results = []
        for qa_idx, qa in enumerate(sample['qa']):
            category = qa['category']
            
            if int(category) == 5 or category not in allow_categories:
                continue
            
            question = qa['question']
            reference = qa['answer']
            
            logger.info(f"\nQuestion {qa_idx+1} (Category {category})")
            logger.info(f"Q: {question}")
            logger.info(f"A: {reference}")
            
            time_start = time.time()
            retrieved_entries = retrieve_by_speaker(
                entries, retriever, question, limit_per_speaker, logger
            )
            retrieval_time = time.time() - time_start
            
            if not retrieved_entries:
                logger.warning(f"No entries retrieved")
                qa_results.append({
                    'question': question,
                    'prediction': '',
                    'reference': reference,
                    'category': category,
                    'retrieved_count': 0,
                    'retrieval_time': retrieval_time,
                    'speaker_distribution': {},
                    'gpu_id': gpu_id,
                    'error': 'No entries retrieved',
                    'metrics': {}
                })
                continue
            
            speaker_dist = {}
            for entry in retrieved_entries:
                speaker = entry.get('_retrieved_speaker', 'Unknown')
                speaker_dist[speaker] = speaker_dist.get(speaker, 0) + 1
            
            logger.info(
                f"Retrieved {len(retrieved_entries)} entries in {retrieval_time:.3f}s"
            )
            logger.info(f"Speaker distribution: {speaker_dist}")
            
            user_prompt = build_prompt_with_speaker_memories(question, retrieved_entries)
            
            try:
                openai_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
                
                response = openai_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": user_prompt}
                    ],
                    temperature=0.0
                )
                generated_answer = response.choices[0].message.content
                logger.info(f"Generated: {generated_answer}")
            except Exception as e:
                logger.error(f"Failed to generate answer: {e}", exc_info=True)
                generated_answer = ""
            
            try:
                label = evaluate_llm_judge(
                    question, reference, generated_answer, 
                    client_obj=llm.client, model_name=JUDGE_MODEL
                )
                metrics = {
                    'judge_correct': int(label),
                    'judge_response': 'CORRECT' if int(label) == 1 else 'WRONG'
                }
                logger.info(
                    f"Judge: {'CORRECT' if int(label) == 1 else 'WRONG'}"
                )
            except Exception as e:
                logger.error(f"Judge failed: {e}", exc_info=True)
                metrics = {'judge_correct': 0, 'judge_response': ''}
            
            qa_results.append({
                'question': question,
                'prediction': generated_answer,
                'reference': reference,
                'category': category,
                'retrieved_count': len(retrieved_entries),
                'speaker_distribution': speaker_dist,
                'retrieval_time': retrieval_time,
                'gpu_id': gpu_id,
                'metrics': metrics
            })
        
        result = {
            'sample_id': sample_id, 
            'gpu_id': gpu_id,
            'results': qa_results, 
            'status': 'success'
        }
        
        sample_file = os.path.join(output_dir, f"sample_{sample_id}.json")
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✓ Sample {sample_id} completed on GPU {gpu_id}: {len(qa_results)} QAs processed")
        logger.info(f"{'='*80}\n")
        
        return result
        
    except Exception as e:
        logger.error(f"✗ Sample {sample_id} failed on GPU {gpu_id}: {str(e)}", exc_info=True)
        return {
            'sample_id': sample_id,
            'gpu_id': gpu_id,
            'status': 'failed',
            'error': str(e),
            'results': []
        }


def check_completed_samples(output_dir: str, samples: List[Dict]) -> tuple:
    completed = []
    missing = []
    
    for sample in samples:
        sample_id = sample['sample_id']
        sample_file = os.path.join(output_dir, f"sample_{sample_id}.json")
        
        if os.path.exists(sample_file):
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                if result.get('status') == 'success' and result.get('results'):
                    completed.append(sample_id)
                    continue
            except:
                pass
        
        missing.append(sample)
    
    return completed, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=DATA_PATH)
    parser.add_argument('--qdrant-dir', type=str, default=QDRANT_DATA_DIR)
    parser.add_argument('--output-dir', type=str, default=RESULTS_DIR)
    parser.add_argument('--limit-per-speaker', type=int, default=RETRIEVAL_LIMIT_PER_SPEAKER)
    parser.add_argument('--allow-categories', type=int, nargs='+', 
                       default=[1, 2, 3, 4])
    parser.add_argument('--embedder', type=str, choices=['huggingface', 'openai'], 
                       default='huggingface')
    parser.add_argument('--gpus', type=int, nargs='+', default=AVAILABLE_GPUS)
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS)

    args = parser.parse_args()
    
    main_logger = logging.getLogger("vector_eval.main")
    main_logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(os.path.join(RUN_LOG_DIR, "main.log"), mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    main_logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    main_logger.addHandler(ch)
    
    main_logger.info("=" * 80)
    main_logger.info("Vector Baseline Evaluation - GPU Balanced Parallel Version")
    main_logger.info(f"Config:")
    main_logger.info(f"  - Dataset: {args.dataset}")
    main_logger.info(f"  - Qdrant dir: {args.qdrant_dir}")
    main_logger.info(f"  - Output dir: {args.output_dir}")
    main_logger.info(f"  - Limit per speaker: {args.limit_per_speaker}")
    main_logger.info(f"  - Allow categories: {args.allow_categories}")
    main_logger.info(f"  - Embedder: {args.embedder}")
    main_logger.info(f"  - Available GPUs: {args.gpus}")
    main_logger.info(f"  - Max workers: {args.max_workers}")
    main_logger.info(f"  - Executor: ProcessPoolExecutor")
    main_logger.info("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main_logger.info(f"\nLoading dataset from {args.dataset}")
    samples = parse_locomo_dataset(args.dataset)
    main_logger.info(f"Loaded {len(samples)} samples")
    
    main_logger.info("\nScanning for completed samples...")
    completed, missing = check_completed_samples(args.output_dir, samples)
    
    main_logger.info(f"✓ Completed: {len(completed)}/{len(samples)}")
    for sid in completed:
        main_logger.info(f"  - {sid}")
    
    main_logger.info(f"✗ Need to process: {len(missing)}/{len(samples)}")
    for sample in missing:
        main_logger.info(f"  - {sample['sample_id']}")
    
    if not missing:
        main_logger.info("\nAll samples already processed!")
    else:
        main_logger.info(f"\n{'='*80}")
        main_logger.info("GPU Assignment Plan:")
        main_logger.info(f"{'='*80}")
        
        gpu_assignments = {}
        for i, sample in enumerate(missing):
            gpu_id = get_gpu_for_sample(sample['sample_id'], args.gpus)
            gpu_assignments[sample['sample_id']] = gpu_id
            main_logger.info(f"  {sample['sample_id']} → GPU {gpu_id}")
        
        main_logger.info(f"\n{'='*80}")
        main_logger.info(f"Starting parallel processing with {args.max_workers} workers...")
        main_logger.info(f"{'='*80}\n")
        
        start_time = time.time()
        all_results = []
        failed_samples = []
        
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_sample = {
                executor.submit(
                    process_single_sample_with_gpu,
                    sample,
                    gpu_assignments[sample['sample_id']],
                    args.embedder,
                    args.qdrant_dir,
                    args.allow_categories,
                    args.limit_per_speaker,
                    args.output_dir
                ): sample
                for sample in missing
            }
            
            with tqdm(total=len(missing), desc="Processing samples") as pbar:
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                        
                        if result.get('status') == 'success':
                            pbar.set_postfix_str(
                                f"✓ {result['sample_id']} (GPU {result.get('gpu_id', '?')})"
                            )
                        else:
                            failed_samples.append(result['sample_id'])
                            pbar.set_postfix_str(f"✗ {result['sample_id']}")
                            
                    except Exception as e:
                        main_logger.error(
                            f"Unexpected error for {sample['sample_id']}: {e}",
                            exc_info=True
                        )
                        failed_samples.append(sample['sample_id'])
                    
                    pbar.update(1)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        gpu_stats = {}
        for result in all_results:
            if result.get('status') == 'success':
                gpu_id = result.get('gpu_id', 'unknown')
                if gpu_id not in gpu_stats:
                    gpu_stats[gpu_id] = 0
                gpu_stats[gpu_id] += 1
        
        main_logger.info(f"\n{'='*80}")
        main_logger.info("Parallel processing completed!")
        main_logger.info(f"Total time: {total_duration:.2f}s ({total_duration/60:.2f} min)")
        main_logger.info(f"Successful: {len(all_results) - len(failed_samples)}")
        main_logger.info(f"Failed: {len(failed_samples)}")
        
        main_logger.info(f"\nGPU Usage Statistics:")
        for gpu_id in sorted(gpu_stats.keys()):
            count = gpu_stats[gpu_id]
            main_logger.info(f"  GPU {gpu_id}: {count} samples")
        
        if failed_samples:
            main_logger.info("\nFailed samples:")
            for sid in failed_samples:
                main_logger.info(f"  - {sid}")
        main_logger.info(f"{'='*80}\n")
    
    main_logger.info("Aggregating all results...")
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    
    for sample in samples:
        sample_file = os.path.join(args.output_dir, f"sample_{sample['sample_id']}.json")
        if os.path.exists(sample_file):
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                for qa_result in result.get('results', []):
                    total_questions += 1
                    category = qa_result['category']
                    category_counts[category] += 1
                    all_metrics.append(qa_result['metrics'])
                    all_categories.append(category)
            except Exception as e:
                main_logger.error(
                    f"Failed to load result for {sample['sample_id']}: {e}"
                )
    
    main_logger.info("Calculating aggregate metrics...")
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
    
    final_results = {
        "model": LLM_MODEL,
        "dataset": args.dataset,
        "total_questions": total_questions,
        "total_samples": len(samples),
        "category_distribution": {str(cat): count for cat, count in category_counts.items()},
        "config": {
            "limit_per_speaker": args.limit_per_speaker,
            "embedding_model": EMBEDDING_MODEL_PATH if args.embedder == 'huggingface' else 'text-embedding-3-small',
            "method": "speaker_based_vector_retrieval_gpu_balanced",
            "allow_categories": args.allow_categories,
            "available_gpus": args.gpus,
            "max_workers": args.max_workers,
            "executor_type": "ProcessPool"
        },
        "aggregate_metrics": aggregate_results,
        "timestamp": RUN_TIMESTAMP
    }
    
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    main_logger.info("\n" + "=" * 80)
    main_logger.info("Evaluation completed!")
    main_logger.info(f"Total samples: {len(samples)}")
    main_logger.info(f"Total questions: {total_questions}")
    main_logger.info("\nCategory Distribution:")
    for category, count in sorted(category_counts.items()):
        if count > 0:
            main_logger.info(
                f"  Category {category}: {count} questions "
                f"({count/total_questions*100:.1f}%)"
            )
    
    main_logger.info("\nAggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        main_logger.info(f"\n{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            if isinstance(stats, dict):
                main_logger.info(f"  {metric_name}:")
                for stat_name, value in stats.items():
                    main_logger.info(f"    {stat_name}: {value:.4f}")
    
    main_logger.info(f"\nResults saved to: {args.output_dir}")
    main_logger.info(f"Logs saved to: {RUN_LOG_DIR}")
    main_logger.info("=" * 80)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()