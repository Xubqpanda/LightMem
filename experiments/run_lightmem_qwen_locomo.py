from openai import OpenAI
import json
from tqdm import tqdm
import datetime
import time
import os
import logging
from lightmem.memory.lightmem import LightMemory
from lightmem.factory.retriever.embeddingretriever.factory import EmbeddingRetrieverFactory
from lightmem.configs.retriever.embeddingretriever.base import EmbeddingRetrieverConfig
from lightmem.factory.retriever.embeddingretriever.qdrant import Qdrant
from lightmem.configs.retriever.embeddingretriever.qdrant import QdrantConfig
import sqlite3
import shutil

# Script-level logger (handlers configured by LightMemory logging config).
script_logger = logging.getLogger("lightmem.experiments.run_lightmem_qwen")
script_logger.setLevel(logging.INFO)

# ============ logging Configuration ============
LOGS_ROOT = "./logs"
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(LOGS_ROOT, RUN_TIMESTAMP)
os.makedirs(RUN_LOG_DIR, exist_ok=True)

# ============ API Configuration ============
API_KEY='sk-elh8U89D8GUNipT9CdDaF39a1e5d44649e48B392E98cC24d'
API_BASE_URL='https://api.gpts.vin/v1'
LLM_MODEL='gpt-4o-mini'
JUDGE_MODEL='gpt-4o-mini'

# ============ Model Paths ============
LLMLINGUA_MODEL_PATH='/disk/disk_20T/fangjizhan/models/llmlingua-2-bert-base-multilingual-cased-meetingbank'

# ============ Data Configuration ============
DATA_PATH='/disk/disk_20T/xubuqiang/lightmem/dataset/locomo/locomo10.json'
DATASET_TYPE='locomo'

# ============ Qdrant 数据目录配置 ============
# 两个独立的数据目录
QDRANT_PRE_UPDATE_DIR='./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_512_0_8_pre_update_v5'   # update前的备份数据
QDRANT_POST_UPDATE_DIR='./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_512_0_8_post_update_v5'  # 工作目录（包含update后的数据）

# 确保目录存在
os.makedirs(QDRANT_PRE_UPDATE_DIR, exist_ok=True)
os.makedirs(QDRANT_POST_UPDATE_DIR, exist_ok=True)
# =============================================

def parse_locomo_timestamp(timestamp_str):
    """将 LoCoMo 时间戳转换为标准格式"""
    timestamp_str = timestamp_str.strip("()")
    try:
        dt = datetime.datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        script_logger.warning(f"Failed to parse timestamp: {timestamp_str}")
        return timestamp_str

def extract_locomo_sessions(conversation_dict):
    """从 LoCoMo conversation 提取 sessions"""
    speaker_a = conversation_dict.get('speaker_a', 'Speaker_A')
    speaker_b = conversation_dict.get('speaker_b', 'Speaker_B')
    
    session_nums = set()
    for key in conversation_dict.keys():
        if key.startswith('session_') and not key.endswith('_date_time'):
            try:
                num = int(key.split('_')[1])
                session_nums.add(num)
            except:
                continue
    
    sessions = []
    timestamps = []
    
    for num in sorted(session_nums):
        session_key = f'session_{num}'
        timestamp_key = f'{session_key}_date_time'
        
        if session_key not in conversation_dict:
            continue
            
        session_data = conversation_dict[session_key]
        timestamp = conversation_dict.get(timestamp_key, '')
        
        messages = []
        for turn in session_data:
            speaker_name = turn['speaker']
            speaker_id = 'speaker_a' if speaker_name == speaker_a else 'speaker_b'
            content = turn['text']
            messages.append({
                "role": "user",
                "content": content,
                "speaker_id": speaker_id,  
                "speaker_name": speaker_name,  
            })
            messages.append({
                "role": "assistant",
                "content": "",
                "speaker_id": speaker_id,  
                "speaker_name": speaker_name,
            })
        
        sessions.append(messages)
        timestamps.append(parse_locomo_timestamp(timestamp))
    
    return sessions, timestamps, speaker_a, speaker_b

class LLMModel:
    def __init__(self, model_name, api_key, base_url):
        self.name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = 0.0
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def call(self, messages: list, **kwargs):
        max_retries = kwargs.get("max_retries", 3)
    
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stream=False
                )
                response = completion.choices[0].message.content
                script_logger.info(response)
                return response
            except Exception as e:
                if attempt == max_retries - 1:
                    raise


def load_lightmem(collection_name):
    """
    加载LightMemory实例（总是使用POST_UPDATE目录作为工作目录）
    """
    config = {
        "pre_compress": True,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": LLMLINGUA_MODEL_PATH,
                    "device_map": "cuda",
                    "use_llmlingua2": True,
                },
                "compress_config": {
                    "instruction": "",
                    "rate": 0.8,
                    "target_token": -1
                },
            }
        },
        "topic_segment": True,
        "precomp_topic_shared": True,
        "topic_segmenter": {
            "model_name": "llmlingua-2",
        },
        "messages_use": "user_only",
        "metadata_generate": True,
        "text_summary": True,
        "memory_manager": {
            "model_name": "openai",
            "configs": {
                "model": LLM_MODEL,
                "api_key": API_KEY,
                "max_tokens": 16000,
                "openai_base_url": API_BASE_URL
            },
        },
        "extract_threshold": 0.1,
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "openai",
            "configs": {
                "model": "text-embedding-3-small",
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": collection_name,
                "embedding_model_dims": 1536,
                "path": f'{QDRANT_POST_UPDATE_DIR}/{collection_name}',  # 始终使用POST目录
            }
        },
        "update": "offline",
        "logging": {
            "level": "DEBUG",
            "file_enabled": True,
            "log_dir": RUN_LOG_DIR,
        }
    }
    
    lightmem = LightMemory.from_config(config)
    return lightmem


llm_judge = LLMModel(JUDGE_MODEL, API_KEY, API_BASE_URL)
llm = LLMModel(LLM_MODEL, API_KEY, API_BASE_URL)

# 加载数据
data = json.load(open(DATA_PATH, "r"))

script_logger.info(f"Loaded {len(data)} samples from LoCoMo dataset")


def collection_entry_count(collection_name: str, base_dir: str) -> int:
    """
    返回指定collection中的entry数量
    
    Args:
        collection_name: collection名称
        base_dir: 基础数据目录
    """
    try:
        cfg = QdrantConfig(
            collection_name=collection_name,
            path=base_dir,
            embedding_model_dims=1536,
            on_disk=True,
        )
        q = Qdrant(cfg)
        try:
            points = q.get_all(with_vectors=False, with_payload=False)
            if points:
                return len(points)
        except Exception as e:
            script_logger.debug(f"Qdrant get_all failed for {collection_name}: {e}")

        # Fallback: try reading the sqlite storage directly
        storage_sqlite = os.path.join(
            base_dir, collection_name, 'collection', collection_name, 'storage.sqlite'
        )
        if not os.path.exists(storage_sqlite):
            return 0

        try:
            conn = sqlite3.connect(storage_sqlite)
            cur = conn.execute("SELECT count(*) FROM points")
            row = cur.fetchone()
            conn.close()
            if row:
                return int(row[0])
            return 0
        except Exception as e:
            script_logger.warning(f"Failed to read storage.sqlite for {collection_name}: {e}")
            return -1
    except Exception as e:
        script_logger.warning(f"Failed to init local Qdrant client for {collection_name}: {e}")
        storage_sqlite = os.path.join(
            base_dir, collection_name, 'collection', collection_name, 'storage.sqlite'
        )
        if os.path.exists(storage_sqlite):
            try:
                conn = sqlite3.connect(storage_sqlite)
                cur = conn.execute("SELECT count(*) FROM points")
                row = cur.fetchone()
                conn.close()
                if row:
                    return int(row[0])
                return 0
            except Exception as e2:
                script_logger.warning(f"Fallback sqlite read also failed for {collection_name}: {e2}")
                return -1
        return -1


# ============ 阶段 1: 构建 Memory ============
script_logger.info("=" * 70)
script_logger.info("Phase 1: Building Memory Collections")
script_logger.info("=" * 70)
script_logger.info(f"Working directory (post-update): {QDRANT_POST_UPDATE_DIR}")
script_logger.info(f"Backup directory (pre-update):   {QDRANT_PRE_UPDATE_DIR}")
script_logger.info("=" * 70)

# Phase 1a: 扫描已完成的samples
script_logger.info("\nScanning existing collections...")
missing = []
for sample in data:
    sample_id = sample['sample_id']
    
    # 检查两个目录
    pre_update_dir = f'{QDRANT_PRE_UPDATE_DIR}/{sample_id}'
    post_update_dir = f'{QDRANT_POST_UPDATE_DIR}/{sample_id}'
    
    pre_exists = os.path.exists(pre_update_dir)
    post_exists = os.path.exists(post_update_dir)
    
    # 如果两个目录都存在且都有数据，则跳过
    if pre_exists and post_exists:
        pre_count = collection_entry_count(sample_id, QDRANT_PRE_UPDATE_DIR)
        post_count = collection_entry_count(sample_id, QDRANT_POST_UPDATE_DIR)
        
        if pre_count > 0 and post_count > 0:
            script_logger.info(
                f"✓ {sample_id}: COMPLETED "
                f"(pre_update={pre_count}, post_update={post_count})"
            )
            continue
    
    # 否则需要重新构建
    status = []
    if not pre_exists:
        status.append("pre_update_missing")
    elif collection_entry_count(sample_id, QDRANT_PRE_UPDATE_DIR) <= 0:
        status.append("pre_update_empty")
        
    if not post_exists:
        status.append("post_update_missing")
    elif collection_entry_count(sample_id, QDRANT_POST_UPDATE_DIR) <= 0:
        status.append("post_update_empty")
    
    script_logger.info(f"✗ {sample_id}: NEEDS BUILD ({', '.join(status)})")
    missing.append(sample)

script_logger.info(f"\nScan complete: {len(missing)}/{len(data)} samples need building\n")

# Phase 1b: 构建缺失的samples
for sample in tqdm(missing, desc="Building memories"):
    sample_id = sample['sample_id']
    script_logger.info(f"\n{'='*70}")
    script_logger.info(f"Building memory for sample: {sample_id}")
    script_logger.info(f"{'='*70}")

    # 提取 sessions
    conversation = sample['conversation']
    sessions, timestamps, speaker_a, speaker_b = extract_locomo_sessions(conversation)

    script_logger.info(f"  Sessions: {len(sessions)}")
    script_logger.info(f"  Speakers: {speaker_a}, {speaker_b}")

    # ====== 步骤1: 在POST目录初始化并构建memory ======
    script_logger.info(f"\n{'─'*70}")
    script_logger.info("Step 1: Building memory (add_memory phase)")
    script_logger.info(f"{'─'*70}")
    
    # 初始化lightmem（使用POST目录）
    lightmem = load_lightmem(collection_name=sample_id)
    
    initial_stats = lightmem.get_token_statistics()
    case_start_time = time.time()
    add_memory_start_time = time.time()
    
    initial_add_tokens = initial_stats['llm']['add_memory']['total_tokens']
    initial_add_calls = initial_stats['llm']['add_memory']['calls']
    
    for session, timestamp in zip(sessions, timestamps):
        while session and session[0]["role"] != "user":
            session.pop(0)
        num_turns = len(session) // 2  
        for turn_idx in range(num_turns):
            turn_messages = session[turn_idx*2 : turn_idx*2 + 2]
            if len(turn_messages) < 2 or turn_messages[0]["role"] != "user" or turn_messages[1]["role"] != "assistant":
                continue
            for msg in turn_messages:
                msg["time_stamp"] = timestamp
            is_last_turn = (
                session is sessions[-1] and turn_idx == num_turns - 1
            )
            result = lightmem.add_memory(
                messages=turn_messages,
                force_segment=is_last_turn,
                force_extract=is_last_turn,
            )
    
    add_memory_end_time = time.time()
    add_memory_duration = add_memory_end_time - add_memory_start_time
    
    # 获取add_memory的token统计
    add_memory_stats = lightmem.get_token_statistics()
    case_add_tokens = add_memory_stats['llm']['add_memory']['total_tokens'] - initial_add_tokens
    case_add_calls = add_memory_stats['llm']['add_memory']['calls'] - initial_add_calls
    case_add_prompt = add_memory_stats['llm']['add_memory']['prompt_tokens'] - initial_stats['llm']['add_memory']['prompt_tokens']
    case_add_completion = add_memory_stats['llm']['add_memory']['completion_tokens'] - initial_stats['llm']['add_memory']['completion_tokens']
    
    after_add_count = collection_entry_count(sample_id, QDRANT_POST_UPDATE_DIR)
    
    script_logger.info(f"✓ Add_memory phase completed: {after_add_count} entries in {add_memory_duration:.2f}s")
    
    # ====== 步骤2: 备份到PRE目录（update前的状态） ======
    script_logger.info(f"\n{'─'*70}")
    script_logger.info("Step 2: Backing up pre-update state")
    script_logger.info(f"{'─'*70}")
    
    source_dir = f'{QDRANT_POST_UPDATE_DIR}/{sample_id}'
    backup_dir = f'{QDRANT_PRE_UPDATE_DIR}/{sample_id}'
    
    backup_start_time = time.time()
    
    # 如果备份目录已存在，先删除
    if os.path.exists(backup_dir):
        script_logger.info(f"  Removing existing backup directory...")
        shutil.rmtree(backup_dir)
    
    # 复制整个collection目录作为备份
    script_logger.info(f"  Copying: {source_dir} -> {backup_dir}")
    shutil.copytree(source_dir, backup_dir)
    
    backup_end_time = time.time()
    backup_duration = backup_end_time - backup_start_time
    
    # 验证备份
    pre_update_count = collection_entry_count(sample_id, QDRANT_PRE_UPDATE_DIR)
    script_logger.info(f"✓ Backup completed: {pre_update_count} entries in {backup_duration:.2f}s")
    
    # ====== 步骤3: 在同一个lightmem实例上执行update ======
    script_logger.info(f"\n{'─'*70}")
    script_logger.info("Step 3: Performing update (offline_update phase)")
    script_logger.info(f"{'─'*70}")
    
    # 记录update前的token状态
    update_start_stats = lightmem.get_token_statistics()
    initial_update_tokens = update_start_stats['llm']['update']['total_tokens']
    initial_update_calls = update_start_stats['llm']['update']['calls']
    
    update_start_time = time.time()
    lightmem.construct_update_queue_all_entries()
    lightmem.offline_update_all_entries(score_threshold=0.8)
    update_end_time = time.time()
    update_duration = update_end_time - update_start_time
    
    # 获取update的token统计
    update_end_stats = lightmem.get_token_statistics()
    case_update_tokens = update_end_stats['llm']['update']['total_tokens'] - initial_update_tokens
    case_update_calls = update_end_stats['llm']['update']['calls'] - initial_update_calls
    case_update_prompt = update_end_stats['llm']['update']['prompt_tokens'] - update_start_stats['llm']['update']['prompt_tokens']
    case_update_completion = update_end_stats['llm']['update']['completion_tokens'] - update_start_stats['llm']['update']['completion_tokens']
    
    post_update_count = collection_entry_count(sample_id, QDRANT_POST_UPDATE_DIR)
    
    script_logger.info(f"✓ Update completed: {post_update_count} entries in {update_duration:.2f}s")
    
    case_end_time = time.time()
    case_total_duration = case_end_time - case_start_time
    
    # ====== 输出统计信息 ======
    script_logger.info(f"\n{'='*70}")
    script_logger.info(f"STATISTICS FOR: {sample_id}")
    script_logger.info(f"{'='*70}")
    
    script_logger.info(f"\n【Directory Information】")
    script_logger.info(f"  Pre-update (backup): {QDRANT_PRE_UPDATE_DIR}/{sample_id} ({pre_update_count} entries)")
    script_logger.info(f"  Post-update (final): {QDRANT_POST_UPDATE_DIR}/{sample_id} ({post_update_count} entries)")
    script_logger.info(f"  Entry change:        {post_update_count - pre_update_count:+d}")
    
    script_logger.info(f"\n【Time Statistics】")
    script_logger.info(f"  Total time:        {case_total_duration:.2f}s (100.0%)")
    script_logger.info(f"  ├─ Add_memory:     {add_memory_duration:.2f}s ({add_memory_duration/case_total_duration*100:.1f}%)")
    script_logger.info(f"  ├─ Backup:         {backup_duration:.2f}s ({backup_duration/case_total_duration*100:.1f}%)")
    script_logger.info(f"  └─ Update:         {update_duration:.2f}s ({update_duration/case_total_duration*100:.1f}%)")

    script_logger.info(f"\n【Add_Memory Token Statistics】")
    script_logger.info(f"  API calls:         {case_add_calls}")
    script_logger.info(f"  Prompt tokens:     {case_add_prompt:,}")
    script_logger.info(f"  Completion tokens: {case_add_completion:,}")
    script_logger.info(f"  Total tokens:      {case_add_tokens:,}")

    script_logger.info(f"\n【Update Token Statistics】")
    script_logger.info(f"  API calls:         {case_update_calls}")
    script_logger.info(f"  Prompt tokens:     {case_update_prompt:,}")
    script_logger.info(f"  Completion tokens: {case_update_completion:,}")
    script_logger.info(f"  Total tokens:      {case_update_tokens:,}")

    script_logger.info(f"\n【Total Token Usage】")
    script_logger.info(f"  Total API calls:   {case_add_calls + case_update_calls}")
    script_logger.info(f"  Total tokens:      {case_add_tokens + case_update_tokens:,}")
    script_logger.info(f"{'='*70}\n")

script_logger.info("\n" + "="*70)
script_logger.info("Phase 1 COMPLETED: All memories built")
script_logger.info(f"Pre-update backup:  {QDRANT_PRE_UPDATE_DIR}")
script_logger.info(f"Post-update final:  {QDRANT_POST_UPDATE_DIR}")
script_logger.info("="*70)