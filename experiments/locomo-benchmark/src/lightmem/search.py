
import os
import sys
import json
import time
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Ensure the experiments directory is importable so we can reuse retrievers
EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if EXPERIMENTS_DIR not in sys.path:
	sys.path.insert(0, EXPERIMENTS_DIR)

try:
	from retrievers import QdrantEntryLoader, VectorRetriever, format_related_memories, LLMModel
except Exception as e:
	QdrantEntryLoader = None
	VectorRetriever = None
	format_related_memories = None
	LLMModel = None
	logger.warning(f"Failed to import retrievers helpers: {e}")


class HFEmbedder:
	def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
		try:
			from sentence_transformers import SentenceTransformer
			self.model = SentenceTransformer(model_name)
		except Exception as e:
			raise RuntimeError(f"Failed to load SentenceTransformer model {model_name}: {e}")

	def embed(self, text: str) -> List[float]:
		vec = self.model.encode(text)
		return vec.tolist() if hasattr(vec, 'tolist') else list(vec)


class OpenAIEmbedder:
	def __init__(self, api_key: str = None, api_base: str = None, model: str = 'text-embedding-3-small'):
		try:
			from openai import OpenAI
			self.client = OpenAI(api_key=api_key, base_url=api_base)
			self.model = model
		except Exception as e:
			raise RuntimeError(f"Failed to init OpenAI client for embeddings: {e}")

	def embed(self, text: str) -> List[float]:
		resp = self.client.embeddings.create(model=self.model, input=text)
		vec = resp.data[0].embedding
		return list(vec)


class LightMemSearch:
	"""Minimal LightMem search wrapper for the locomo benchmark.

	Usage:
		s = LightMemSearch(qdrant_dir='./qdrant_data_locomo', embedder_type='huggingface')
		s.process_data_file('dataset/locomo10.json', output_file='results/lightmem_search.json')
	"""

	def __init__(self, qdrant_dir: str = './qdrant_data_locomo', embedder_type: str = 'openai',
				 embedding_model_path: str = None, api_key: str = None, api_base: str = None, top_k: int = 30):
		self.qdrant_dir = qdrant_dir
		self.top_k = top_k
		self.api_key = api_key or os.environ.get('OPENAI_API_KEY') or os.environ.get('LIGHTMEM_OPENAI_API_KEY')
		self.api_base = api_base or os.environ.get('OPENAI_API_BASE') or os.environ.get('LIGHTMEM_OPENAI_BASE')
		emb_path = embedding_model_path or os.environ.get('EMBEDDING_MODEL_PATH', 'sentence-transformers/all-MiniLM-L6-v2')
		self.embedder = None
		if embedder_type == 'huggingface':
			try:
				self.embedder = HFEmbedder(model_name=emb_path)
			except Exception as e:
				logger.warning(f"HF embedder init failed, falling back to OpenAI: {e}")
				self.embedder = OpenAIEmbedder(api_key=self.api_key, api_base=self.api_base)
		else:
			try:
				self.embedder = OpenAIEmbedder(api_key=self.api_key, api_base=self.api_base)
			except Exception as e:
				logger.warning(f"OpenAI embedder init failed, trying HF: {e}")
				self.embedder = HFEmbedder(model_name=emb_path)

		if QdrantEntryLoader is None or VectorRetriever is None:
			logger.error("Required retrievers helpers are not available. Make sure experiments/retrievers.py is on PYTHONPATH.")

		self.entry_loader = QdrantEntryLoader(self.qdrant_dir) if QdrantEntryLoader is not None else None
		self.retriever = VectorRetriever(self.embedder) if VectorRetriever is not None else None

	def process_data_file(self, data_path: str = 'dataset/locomo10.json', output_file: str = None):
		data = json.load(open(data_path, 'r', encoding='utf-8'))
		results = []
		out_path = output_file or os.path.join('results', 'lightmem_search_results.json')
		os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

		for sample in data:
			sample_id = sample.get('sample_id')
			logger.info(f"Processing sample {sample_id}")
			if self.entry_loader is None:
				logger.error("No entry loader available, skipping retrieval")
				break
			entries = self.entry_loader.load_entries(sample_id, with_vectors=True)
			if not entries:
				logger.warning(f"No entries found for {sample_id}, skipping")
				continue

			# build id->entry mapping
			id_map = {str(e.get('id')): e for e in entries}

			sample_results = {'sample_id': sample_id, 'results': []}
			for qa in sample.get('qa', []):
				category = int(qa.get('category', 0))
				if category == 5:
					# skip adversarial category to match other baselines
					continue
				question = qa.get('question', '')
				reference = qa.get('answer', '')
				t0 = time.time()
				retrieved = []
				try:
					retrieved = self.retriever.retrieve(entries, question, limit=self.top_k)
				except Exception as e:
					logger.warning(f"Retrieval failed for {sample_id}: {e}")
					retrieved = []
				t1 = time.time()

				# enrich retrieved with payload / original entry for formatting
				enriched = []
				for r in retrieved:
					rid = str(r.get('id'))
					ent = id_map.get(rid) or {}
					enriched.append({'id': rid, 'score': r.get('score'), 'payload': ent.get('payload', {}), 'entry': ent})

				formatted = format_related_memories(enriched) if format_related_memories is not None else ''

				qa_result = {
					'question': question,
					'reference': reference,
					'category': category,
					'prediction_context': formatted,
					'retrieved_count': len(enriched),
					'retrieval_time': t1 - t0,
					'retrieved': [{'id': e['id'], 'score': e['score']} for e in enriched]
				}
				sample_results['results'].append(qa_result)

			results.append(sample_results)

		with open(out_path, 'w', encoding='utf-8') as f:
			json.dump({'timestamp': int(time.time()), 'results': results}, f, ensure_ascii=False, indent=2)

		logger.info(f"Search results saved to {out_path}")

