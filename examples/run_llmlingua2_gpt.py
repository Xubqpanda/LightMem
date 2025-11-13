import os
import logging
import sys
from lightmem.memory.lightmem import LightMemory
from lightmem.configs.base import BaseMemoryConfigs

# ============ API Configuration ============
API_KEY='sk-mYmdqXKCUL9FqNfI27855c29E94d419c995bA6D54c20Af21'
API_BASE_URL='https://api.gpts.vin/v1'
LLM_MODEL='qwen3-30b-a3b-instruct-2507'
JUDGE_MODEL='gpt-4o-mini'

# ============ Model Paths ============
LLMLINGUA_MODEL_PATH='/disk/disk_20T/fangjizhan/models/llmlingua-2-bert-base-multilingual-cased-meetingbank'
EMBEDDING_MODEL_PATH='/disk/disk_20T/fangjizhan/models/all-MiniLM-L6-v2'

# ============ Data Configuration ============
DATA_PATH='/disk/disk_20T/xubuqiang/lightmem/dataset/longmemeval/longmemeval_s_cleaned.json'
RESULTS_DIR='../results'
QDRANT_DATA_DIR='./qdrant_data'

config_dict = {
    "pre_compress": True,
    "pre_compressor": {
        "model_name": "llmlingua-2",
        "configs": {
            "llmlingua_config": {
                "model_name": LLMLINGUA_MODEL_PATH,
                "device_map": "cuda",
                "use_llmlingua2": True,
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
        }
    },
    "extract_threshold": 0.1,
    "index_strategy": "embedding",
    "text_embedder": {
        "model_name": "huggingface",
        "configs": {
            "model": EMBEDDING_MODEL_PATH,
            "embedding_dims": 384,
            "model_kwargs": {"device": "cuda"},
        },
    },
    "retrieve_strategy": "embedding",
    "embedding_retriever": {
        "model_name": "qdrant",
        "configs": {
            "collection_name": "my_long_term_chat",
            "embedding_model_dims": 384,
            "path": "./my_long_term_chat", 
        }
    },
    "update": "offline",
    "logging": {
        "level": "DEBUG",
        "file_enabled": True,
        "log_dir": "./logs",
    }
}

lightmem = LightMemory.from_config(config_dict)

### Add Memory
session = {
    "timestamp": "2025-01-10",
    "turns": [
        [
            {"role": "user", "content": "My favorite ice cream flavor is pistachio, and my dog's name is Rex."}, 
            {"role": "assistant", "content": "Got it. Pistachio is a great choice."}
        ], 
    ]
}

for turn_messages in session["turns"]:
    timestamp = session["timestamp"]
    for msg in turn_messages:
        msg["time_stamp"] = timestamp
        
    store_result = lightmem.add_memory(
        messages=turn_messages,
        force_segment=True,
        force_extract=True
    )

### Offline Update
lightmem.construct_update_queue_all_entries()
lightmem.offline_update_all_entries(score_threshold=0.8)

### Retrieve Memory
question = "What is the name of my dog?"
related_memories = lightmem.retrieve(question, limit=5)
print("Related Memories:", related_memories)