import os
import re
import json
import uuid
import tiktoken
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union


@dataclass
class MemoryEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    time_stamp: str = field(default_factory=lambda: datetime.now().isoformat())
    float_time_stamp: float = 0
    weekday: str = ""
    category: str = ""
    subcategory: str = ""
    memory_class: str = ""
    memory: str = ""
    original_memory: str = ""
    compressed_memory: str = ""
    topic_id: Optional[int] = None
    topic_summary: str = ""
    entry_type: str = ""
    speaker_id: str = ""
    speaker_name: str = ""
    hit_time: int = 0
    update_queue: List = field(default_factory=list)

def clean_response(response: str) -> List[Dict[str, Any]]:
    """
    Cleans the model response by:
    1. Removing enclosing code block markers (```[language] ... ```).
    2. Parsing the JSON content safely.
    3. Returning the value of the "data" key if present, otherwise trying to return the parsed list/dict.
    """
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, response.strip())
    cleaned = match.group(1).strip() if match else response.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict) and "data" in parsed and isinstance(parsed["data"], list):
        return parsed["data"]

    if isinstance(parsed, list):
        return parsed

    return []

def assign_sequence_numbers_with_timestamps(extract_list, offset_ms: int = 500, topic_id_mapping: List[List[int]] = None):
    from datetime import datetime, timedelta
    from collections import defaultdict
    
    current_index = 0
    timestamps_list = []
    weekday_list = []
    speaker_list = []
    message_refs = []
    
    for segments in extract_list:
        for seg in segments:
            for message in seg:
                session_time = message.get('session_time', '')
                message_refs.append((message, session_time))
    
    session_groups = defaultdict(list)
    for msg, sess_time in message_refs:
        session_groups[sess_time].append(msg)
    
    for sess_time, messages in session_groups.items():
        base_dt = datetime.strptime(sess_time, "%Y-%m-%d %H:%M:%S")
        for i, msg in enumerate(messages):
            offset = timedelta(milliseconds=offset_ms * i)
            new_dt = base_dt + offset
            msg['time_stamp'] = new_dt.isoformat(timespec='milliseconds')
    
    for segments in extract_list:
        for seg in segments:
            for message in seg:
                message["sequence_number"] = current_index
                timestamps_list.append(message["time_stamp"])
                weekday_list.append(message["weekday"])
                speaker_info = {
                    'speaker_id': message.get('speaker_id', 'unknown'),
                    'speaker_name': message.get('speaker_name', 'Unknown')
                }
                speaker_list.append(speaker_info)
                current_index += 1

    sequence_to_topic = {}
    if topic_id_mapping:
        for api_idx, api_call_segments in enumerate(extract_list):
            for topic_idx, topic_segment in enumerate(api_call_segments):
                tid = topic_id_mapping[api_idx][topic_idx]
                for msg in topic_segment:
                    seq = msg.get("sequence_number")
                    sequence_to_topic[seq] = tid

    return extract_list, timestamps_list, weekday_list, speaker_list, sequence_to_topic

# TODO：merge into context retriever
def save_memory_entries(memory_entries, file_path="memory_entries.json"):
    def entry_to_dict(entry):
        return {
            "id": entry.id,
            "time_stamp": entry.time_stamp,
            "topic_id": entry.topic_id,
            "topic_summary": entry.topic_summary,
            "entry_type": entry.entry_type,
            "category": entry.category,
            "subcategory": entry.subcategory,
            "memory_class": entry.memory_class,
            "memory": entry.memory,
            "original_memory": entry.original_memory,
            "compressed_memory": entry.compressed_memory,
            "hit_time": entry.hit_time,
            "update_queue": entry.update_queue,
        }

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    new_data = [entry_to_dict(e) for e in memory_entries]
    existing_data.extend(new_data)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)


def resolve_tokenizer(tokenizer_or_name: Union[str, Any]):

    if tokenizer_or_name is None:
        raise ValueError("Tokenizer or model_name must be provided.")

    if isinstance(tokenizer_or_name, str):
        model_tokenizer_map = {
            "gpt-4o-mini": "o200k_base",
            "gpt-4o": "o200k_base",
            "gpt-4.1-mini": "o200k_base",
            "gpt-4.1": "o200k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "qwen3-30b-a3b-instruct-2507": "o200k_base"
        }

        if tokenizer_or_name not in model_tokenizer_map:
            raise ValueError(f"Unknown model_name '{tokenizer_or_name}', please update mapping.")

        encoding_name = model_tokenizer_map[tokenizer_or_name]
        print("DEBUG: resolved to encoding", encoding_name)
        return tiktoken.get_encoding(encoding_name)

    raise TypeError(f"Unsupported tokenizer type: {type(tokenizer_or_name)}")


def convert_extraction_results_to_memory_entries(
    extracted_results: List[Optional[Dict]],
    timestamps_list: List,
    weekday_list: List,
    speaker_list: List = None,
    topic_id_map: Dict[int, int] = None,
    logger = None
) -> List[MemoryEntry]:
    """
    Convert extraction results to MemoryEntry objects.

    Args:
        extracted_results: Results from meta_text_extract, each containing cleaned_result
        timestamps_list: List of timestamps indexed by sequence_number
        weekday_list: List of weekdays indexed by sequence_number
        speaker_list: List of speaker information
        topic_id_map: Optional mapping of sequence_number -> topic_id (preferred)
        logger: Optional logger for debug info

    Returns:
        List of MemoryEntry objects with assigned topic_id and timestamps
    """
    memory_entries = []

    extracted_memory_entry = [
        item["cleaned_result"]
        for item in extracted_results
        if item and item.get("cleaned_result")
    ]

    for topic_memory in extracted_memory_entry:
        if not topic_memory:
            continue

        for topic_idx, fact_list in enumerate(topic_memory):
            if not isinstance(fact_list, list):
                fact_list = [fact_list]

            for fact_entry in fact_list:
                sid = int(fact_entry.get("source_id"))
                seq_candidate = sid * 2
                resolved_topic_id = topic_id_map[seq_candidate]
                
                mem_obj = _create_memory_entry_from_fact(
                    fact_entry,
                    timestamps_list,
                    weekday_list,
                    speaker_list,
                    topic_id=resolved_topic_id,
                    topic_summary="",
                    logger=logger,
                )

                if mem_obj:
                    memory_entries.append(mem_obj)

    return memory_entries


def _create_memory_entry_from_fact(
    fact_entry: Dict,
    timestamps_list: List,
    weekday_list: List,
    speaker_list: List = None,
    topic_id: int = None,  
    topic_summary: str = "",
    logger = None
) -> Optional[MemoryEntry]:
    """
    Helper function to create a MemoryEntry from a fact entry.
    
    Args:
        fact_entry: Dict containing source_id and fact
        timestamps_list: List of timestamps indexed by sequence_number
        weekday_list: List of weekdays indexed by sequence_number
        speaker_list: List of speaker information
        topic_id: Topic ID for this memory entry
        topic_summary: Topic summary for this memory entry (reserved for future use)
        logger: Optional logger for warnings
        
    Returns:
        MemoryEntry object or None if creation fails
    """
    sequence_n = fact_entry.get("source_id") * 2
    
    try:
        time_stamp = timestamps_list[sequence_n]
        
        if not isinstance(time_stamp, float):
            from datetime import datetime
            float_time_stamp = datetime.fromisoformat(time_stamp).timestamp()
        else:
            float_time_stamp = time_stamp
            
        weekday = weekday_list[sequence_n]
        speaker_info = speaker_list[sequence_n]
        speaker_id = speaker_info.get('speaker_id', 'unknown')
        speaker_name = speaker_info.get('speaker_name', 'Unknown')
        
    except (IndexError, TypeError, ValueError) as e:
        if logger:
            logger.warning(
                f"Error getting timestamp for sequence {sequence_n}: {e}"
            )
        time_stamp = None
        float_time_stamp = None
        weekday = None
        speaker_id = 'unknown'
        speaker_name = 'Unknown'
    
    entry_type_value = fact_entry.get("entry_type", "")
    if entry_type_value == "interaction":
        memory_content = fact_entry.get("interaction", "")
    else:
        memory_content = fact_entry.get("fact", "")
    
    mem_obj = MemoryEntry(
        time_stamp=time_stamp,
        float_time_stamp=float_time_stamp,
        weekday=weekday,
        memory=memory_content,
        entry_type=entry_type_value,
        speaker_id=speaker_id,
        speaker_name=speaker_name,
        topic_id=topic_id,
        topic_summary=topic_summary,
    )
    
    return mem_obj

def merge_extraction_results(
    factual_results: List[Optional[Dict]], 
    interaction_results: List[Optional[Dict]], 
    logger
) -> List[Optional[Dict]]:
    if interaction_results is None or len(interaction_results) == 0:
        logger.info("No interaction results to merge, returning factual results only")
        return factual_results
    
    if factual_results is None or len(factual_results) == 0:
        logger.info("No factual results, returning interaction results only")
        return interaction_results

    if len(factual_results) != len(interaction_results):
        logger.warning(
            f"Results length mismatch: factual={len(factual_results)}, "
            f"interaction={len(interaction_results)}"
        )
        return factual_results
    
    merged_results = []
    total_interactions = 0
    total_facts = 0
    for batch_idx, (fact_batch, inter_batch) in enumerate(zip(factual_results, interaction_results)):
        if fact_batch is None and inter_batch is None:
            logger.warning(f"Both batches are None at index {batch_idx}")
            merged_results.append(None)
            continue
        
        if fact_batch is None:
            logger.warning(f"Factual batch is None at index {batch_idx}, using interaction only")
            merged_results.append(inter_batch)
            continue
        
        if inter_batch is None:
            logger.debug(f"Interaction batch is None at index {batch_idx}, using factual only")
            merged_results.append(fact_batch)
            continue
        
        fact_usage = fact_batch.get("usage") if isinstance(fact_batch.get("usage"), dict) else {}
        inter_usage = inter_batch.get("usage") if isinstance(inter_batch.get("usage"), dict) else {}

        merged_usage = {}
        if fact_usage or inter_usage:
            usage_keys = set(fact_usage.keys()) | set(inter_usage.keys())
            for key in usage_keys:
                fact_value = fact_usage.get(key)
                inter_value = inter_usage.get(key)

                if isinstance(fact_value, (int, float)) or isinstance(inter_value, (int, float)):
                    merged_usage[key] = (
                        (fact_value if isinstance(fact_value, (int, float)) else 0)
                        + (inter_value if isinstance(inter_value, (int, float)) else 0)
                    )
                else:
                    merged_usage[key] = inter_value if inter_value is not None else fact_value

        merged_batch = {
            "input_prompt": fact_batch.get("input_prompt", []),
            "output_prompt": fact_batch.get("output_prompt", ""),
            "usage": merged_usage,
            "cleaned_result": []
        }
        
        fact_entries = fact_batch.get("cleaned_result", [])
        inter_entries = inter_batch.get("cleaned_result", [])
        
        for entry in fact_entries:
            if "entry_type" not in entry:
                entry["entry_type"] = "fact"
        
        for entry in inter_entries:
            if "entry_type" not in entry:
                entry["entry_type"] = "interaction"
        
        merged_batch["cleaned_result"].extend(fact_entries)
        merged_batch["cleaned_result"].extend(inter_entries)
        batch_facts = len(fact_entries)
        batch_interactions = len(inter_entries)
        total_facts += batch_facts
        total_interactions += batch_interactions
        
        logger.debug(
            f"Merged batch {batch_idx}: "
            f"{batch_facts} facts (with 'fact' field) + "
            f"{batch_interactions} interactions (with 'interaction' field) = "
            f"{len(merged_batch['cleaned_result'])} total (all with 'entry_type')"
        )
        
        merged_results.append(merged_batch)
    
    logger.info(
        f"Merge completed: {len(merged_results)} batches, "
        f"{total_facts} facts, {total_interactions} interactions, "
        f"all entries now have entry_type field"
    )
    
    return merged_results