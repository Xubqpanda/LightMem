#!/usr/bin/env python3
"""
Unified retrieval runner (vector / bipartite / hybrid)

Provides a single entrypoint to run the three retrieval strategies present
in the repo. The goal is to consolidate shared code (LLM wrapper,
Qdrant loader, embedding loader, term extractor, bipartite graph,
vector retriever and hybrid retriever) into one script for easier
maintenance and experimentation.

Usage:
  python run_unified_retrieval.py --mode vector
  python run_unified_retrieval.py --mode bipartite
  python run_unified_retrieval.py --mode hybrid

This file intentionally keeps dependencies optional where reasonable:
spaCy is used when available for term extraction; otherwise a simple
fallback tokenizer is used.
"""

from openai import OpenAI
import argparse
import json
import logging
import os
import time
import datetime
import sqlite3
import pickle
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict, deque

import numpy as np
from tqdm import tqdm
import spacy
SPACY_AVAILABLE = True
from lightmem.factory.text_embedder.huggingface import TextEmbedderHuggingface
from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig
from lightmem.factory.retriever.embeddingretriever.qdrant import Qdrant
from lightmem.configs.retriever.embeddingretriever.qdrant import QdrantConfig

from retrievers import (
    LLMModel,
    QdrantEntryLoader,
    TermExtractor,
    VectorRetriever,
    BipartiteGraph,
    HybridRetriever,
    format_related_memories,
)
# ============ Logging Configuration ==========
LOGS_ROOT = "./logs"
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(LOGS_ROOT, f"unified_retrieval_{RUN_TIMESTAMP}")
os.makedirs(RUN_LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(RUN_LOG_DIR, 'unified_retrieval.log')),
        logging.StreamHandler()
    ]
)
script_logger = logging.getLogger("unified_retrieval")
global RETRIEVAL_LIMIT
# ============ Defaults and Config ============
API_KEY = 'sk-mYmdqXKCUL9FqNfI27855c29E94d419c995bA6D54c20Af21'
API_BASE_URL = 'https://api.gpts.vin/v1'
LLM_MODEL = 'qwen3-30b-a3b-instruct-2507'
JUDGE_MODEL = 'gpt-4o-mini'

DATA_PATH = '/disk/disk_20T/xubuqiang/lightmem/dataset/longmemeval/longmemeval_s_cleaned.json'
QDRANT_DATA_DIR = './qdrant_data_longmemeval'
EMBEDDING_MODEL_PATH = '/disk/disk_20T/fangjizhan/models/all-MiniLM-L6-v2'
SPACY_MODEL = 'en_core_web_sm'

RETRIEVAL_LIMIT = 20
MAX_HOPS = 1
HYBRID_MODE = True
VECTOR_RERANK_ENABLED = True
VECTOR_SUPPLEMENT_ENABLED = True

STOPWORDS = set(x.lower() for x in [
    'user', 'have', 'they', 'it', 'can', 'that', 'make', 'try', 'use', 'be', 'do', 'i', 'you',
    'what', 'when', 'how', 'which', 'who', 'whom', 'whose', 'why',
    'the', 'a', 'an', 'this', 'that', 'these', 'those', 'there', 'here'
])
# Retrieval classes and helpers have been moved into
# lightmem.experiments.retrievers to keep this runner small.


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        return template.format(question, answer, response)


def true_or_false(response: Optional[str]) -> bool:
    if response is None:
        return False
    normalized = str(response).strip().lower()
    if not normalized:
        return False
    first_line = normalized.splitlines()[0].strip()
    tokens = first_line.replace('.', '').replace('!', '').replace(':', '').replace(';', '').split()
    if not tokens:
        return False
    head = tokens[0]
    if head in ('yes', 'y'):
        return True
    if head in ('no', 'n'):
        return False
    if 'yes' in first_line:
        return True
    if 'no' in first_line:
        return False
    return False



def process_item_vector(item: Dict, entry_loader: QdrantEntryLoader, retriever: VectorRetriever, llm: LLMModel, llm_judge: LLMModel) -> Dict:
    qid = item['question_id']
    question = item['question']
    script_logger.info(f"[{qid}] Vector mode processing")
    try:
        entries = entry_loader.load_entries(qid, with_vectors=True)
        if not entries:
            return {'question_id': qid, 'error': 'No entries', 'correct': 0}
    except Exception as e:
        return {'question_id': qid, 'error': str(e), 'correct': 0}

    t0 = time.time()
    retrieved = retriever.retrieve(entries, question, limit=RETRIEVAL_LIMIT)
    retrieval_time = time.time() - t0

    related_memories = []
    for r in retrieved:
        payload = r.get('payload', {})
        mem = payload.get('memory') or payload.get('original_memory') or payload.get('compressed_memory') or ''
        related_memories.append({'id': r['id'], 'memory': mem, 'score': r['score'], 'payload': payload})

    # Format related memories into LightMemory-style strings
    formatted = format_related_memories(related_memories)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Question time:{item.get('question_date')} and question:{question}\nPlease answer the question based on the following memories: {formatted}"}
    ]

    try:
        gen = llm.call(messages)
    except Exception as e:
        return {'question_id': qid, 'retrieved_count': len(retrieved), 'retrieval_time': retrieval_time, 'error': str(e), 'correct': 0}

    prompt = get_anscheck_prompt(item.get('question_type'), question, item.get('answer'), gen, abstention=('abs' in qid))
    try:
        judge = llm_judge.call([{"role": "user", "content": prompt}])
        correct = 1 if true_or_false(judge) else 0
    except Exception as e:
        judge = str(e)
        correct = 0

    return {'question_id': qid, 'question': question, 'retrieved_count': len(retrieved), 'retrieval_time': retrieval_time, 'generated_answer': gen, 'ground_truth': item.get('answer'), 'judge_response': judge, 'correct': correct}


def process_item_bipartite(item: Dict, term_extractor: TermExtractor, entry_loader: QdrantEntryLoader, llm: LLMModel, llm_judge: LLMModel) -> Dict:
    qid = item['question_id']
    question = item['question']
    script_logger.info(f"[{qid}] Bipartite mode processing")
    try:
        entries = entry_loader.load_entries(qid, with_vectors=False)
        if not entries:
            return {'question_id': qid, 'error': 'No entries', 'correct': 0}
    except Exception as e:
        return {'question_id': qid, 'error': str(e), 'correct': 0}

    graph = BipartiteGraph()
    for entry in entries:
        entry_id = str(entry.get('id'))
        payload = entry.get('payload', {}) or {}
        memory_text = payload.get('memory') or payload.get('original_memory') or payload.get('compressed_memory') or payload.get('text') or payload.get('content') or entry.get('memory') or ''
        terms = term_extractor.extract_terms(memory_text)
        topic_id = payload.get('topic_id')
        graph.add_entry(entry_id, terms, payload, topic_id)

    stats = graph.get_stats() if hasattr(graph, 'get_stats') else {}
    query_terms = term_extractor.extract_terms(question)
    if not query_terms:
        query_terms = [w for w in question.lower().split() if w not in STOPWORDS][:5]

    t0 = time.time()
    retrieved = graph.bfs_retrieve(query_terms=query_terms, max_hops=MAX_HOPS, limit=RETRIEVAL_LIMIT)
    retrieval_time = time.time() - t0

    related_memories = []
    for r in retrieved:
        payload = r.get('payload', {})
        mem = payload.get('memory') or payload.get('original_memory') or payload.get('compressed_memory') or ''
        related_memories.append({'id': r['id'], 'memory': mem, 'score': r['score'], 'hop': r.get('hop'), 'payload': payload})

    formatted = format_related_memories(related_memories)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Question time:{item.get('question_date')} and question:{question}\nPlease answer the question based on the following memories: {formatted}"}
    ]

    try:
        gen = llm.call(messages)
    except Exception as e:
        return {'question_id': qid, 'retrieved_count': len(retrieved), 'retrieval_time': retrieval_time, 'error': str(e), 'correct': 0}

    prompt = get_anscheck_prompt(item.get('question_type'), question, item.get('answer'), gen, abstention=('abs' in qid))
    try:
        judge = llm_judge.call([{"role": "user", "content": prompt}])
        correct = 1 if true_or_false(judge) else 0
    except Exception as e:
        judge = str(e)
        correct = 0

    return {'question_id': qid, 'question': question, 'query_terms': query_terms, 'retrieved_count': len(retrieved), 'retrieval_time': retrieval_time, 'generated_answer': gen, 'ground_truth': item.get('answer'), 'judge_response': judge, 'correct': correct, 'graph_stats': stats}


def process_item_hybrid(item: Dict, term_extractor: TermExtractor, entry_loader: QdrantEntryLoader, embedder, llm: LLMModel, llm_judge: LLMModel) -> Dict:
    qid = item['question_id']
    question = item['question']
    script_logger.info(f"[{qid}] Hybrid mode processing")
    try:
        entries = entry_loader.load_entries(qid, with_vectors=False)
        if not entries:
            return {'question_id': qid, 'error': 'No entries', 'correct': 0}
    except Exception as e:
        return {'question_id': qid, 'error': str(e), 'correct': 0}

    graph = BipartiteGraph()
    for entry in entries:
        entry_id = str(entry.get('id'))
        payload = entry.get('payload', {}) or {}
        memory_text = payload.get('memory') or payload.get('original_memory') or payload.get('compressed_memory') or payload.get('text') or payload.get('content') or entry.get('memory') or ''
        terms = term_extractor.extract_terms(memory_text)
        vector = entry.get('vector')
        graph.add_entry(entry_id, terms, payload, payload.get('topic_id'), vector)

    query_terms = term_extractor.extract_terms(question)
    if not query_terms:
        query_terms = [w for w in question.lower().split() if w not in STOPWORDS][:5]

    retriever = HybridRetriever(graph, entry_loader, embedder)
    t0 = time.time()
    retrieved, retrieval_stats = retriever.retrieve(collection_name=qid, query_terms=query_terms, query_text=question, max_hops=MAX_HOPS, limit=RETRIEVAL_LIMIT)
    retrieval_time = time.time() - t0

    related_memories = []
    for r in retrieved:
        payload = r.get('payload', {})
        mem = payload.get('memory') or payload.get('original_memory') or payload.get('compressed_memory') or ''
        related_memories.append({'id': r['id'], 'memory': mem, 'score': r.get('score'), 'hop': r.get('hop'), 'payload': payload})

    formatted = format_related_memories(related_memories)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Question time:{item.get('question_date')} and question:{question}\nPlease answer the question based on the following memories: {formatted}"}
    ]

    try:
        gen = llm.call(messages)
    except Exception as e:
        return {'question_id': qid, 'retrieved_count': len(retrieved), 'retrieval_time': retrieval_time, 'error': str(e), 'correct': 0}

    prompt = get_anscheck_prompt(item.get('question_type'), question, item.get('answer'), gen, abstention=('abs' in qid))
    try:
        judge = llm_judge.call([{"role": "user", "content": prompt}])
        correct = 1 if true_or_false(judge) else 0
    except Exception as e:
        judge = str(e)
        correct = 0

    return {'question_id': qid, 'question': question, 'query_terms': query_terms, 'retrieval_stats': retrieval_stats, 'retrieval_time': retrieval_time, 'generated_answer': gen, 'ground_truth': item.get('answer'), 'judge_response': judge, 'correct': correct}


def main():
    parser = argparse.ArgumentParser(description="Unified retrieval runner: vector|bipartite|hybrid")
    parser.add_argument('--mode', type=str, choices=['vector', 'bipartite', 'hybrid'], default='vector')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of questions to process')
    parser.add_argument('--retrieval-limit', type=int, default=20)
    parser.add_argument('--output-dir', type=str, default=f'./results_unified_{RUN_TIMESTAMP}')
    args = parser.parse_args()

    global RETRIEVAL_LIMIT
    RETRIEVAL_LIMIT = args.retrieval_limit

    script_logger.info("=" * 80)
    script_logger.info(f"Unified retrieval runner (mode={args.mode})")
    script_logger.info("=" * 80)

    entry_loader = QdrantEntryLoader(QDRANT_DATA_DIR)
    term_extractor = TermExtractor(SPACY_MODEL)

    embedder = None
    if TextEmbedderHuggingface is not None and BaseTextEmbedderConfig is not None:
        try:
            cfg = BaseTextEmbedderConfig(model=EMBEDDING_MODEL_PATH, embedding_dims=384, model_kwargs={"device": "cuda"}, huggingface_base_url=None)
            embedder = TextEmbedderHuggingface(cfg)
            script_logger.info("Embedding model loaded")
        except Exception as e:
            script_logger.warning(f"Failed to init embedder: {e}")

    retriever = VectorRetriever(embedder) if embedder is not None else None
    llm = LLMModel(LLM_MODEL, API_KEY, API_BASE_URL)
    llm_judge = LLMModel(JUDGE_MODEL, API_KEY, API_BASE_URL)

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if args.limit:
        data = data[:args.limit]

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    correct_count = 0

    for item in tqdm(data, desc='Processing questions'):
        try:
            if args.mode == 'vector':
                if retriever is None:
                    script_logger.error('Vector mode requested but embedder not available; skipping')
                    res = {'question_id': item['question_id'], 'error': 'embedder not available', 'correct': 0}
                else:
                    res = process_item_vector(item, entry_loader, retriever, llm, llm_judge)
            elif args.mode == 'bipartite':
                res = process_item_bipartite(item, term_extractor, entry_loader, llm, llm_judge)
            else:
                res = process_item_hybrid(item, term_extractor, entry_loader, embedder, llm, llm_judge)
        except Exception as e:
            script_logger.error(f"Processing of item {item.get('question_id')} failed: {e}")
            res = {'question_id': item.get('question_id'), 'error': str(e), 'correct': 0}

        results.append(res)
        if res.get('correct', 0) == 1:
            correct_count += 1

        # save per-question
        out_file = os.path.join(args.output_dir, f"result_{item['question_id']}.json")
        with open(out_file, 'w', encoding='utf-8') as fo:
            json.dump(res, fo, ensure_ascii=False, indent=2)

    accuracy = correct_count / len(results) if results else 0.0
    summary = {'total_questions': len(results), 'correct_count': correct_count, 'accuracy': accuracy, 'mode': args.mode, 'timestamp': RUN_TIMESTAMP}
    with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf-8') as fo:
        json.dump(summary, fo, ensure_ascii=False, indent=2)

    script_logger.info("=" * 80)
    script_logger.info(f"Completed: total={len(results)} correct={correct_count} accuracy={accuracy:.2%}")
    script_logger.info(f"Results saved to: {args.output_dir}")
    script_logger.info("=" * 80)


if __name__ == '__main__':
    main()
