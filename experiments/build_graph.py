import argparse
import json
import re
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists():
    sys.path.append(str(SRC_PATH))

from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig  # type: ignore[import]
from lightmem.factory.text_embedder.openai import TextEmbedderOpenAI  # type: ignore[import]
from retrievers import QdrantEntryLoader, VectorRetriever  # type: ignore[import]

DEFAULT_API_KEY = "sk-elh8U89D8GUNipT9CdDaF39a1e5d44649e48B392E98cC24d"
DEFAULT_BASE_URL = "https://api.gpts.vin/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TIMELINE_MODEL = "gpt-4o-mini"

RETRIEVAL_TOP_K = 5
CONTEXT_WINDOW = 3

TIMELINE_INSTRUCTION = (
    "Identify the event thread that appears most frequently across the entries and focus the response on that storyline. "
    "Describe the real-world sequence in strict chronological order. When an explicit event date/time is given, lead with it in ISO format (YYYY-MM-DD or YYYY-MM-DD HH:MM). When the entry only provides a relative phrase ('yesterday', 'three days later', 'next week'), keep that wording but immediately weave in the mention timestamp as part of the sentence (e.g., 'Three days later on 2024-03-05 18:20' or 'Yesterday at 2024-04-19 10:00'). Avoid inventing specific dates that the entries do not supply, and include the weekday whenever you mention a date. "
    "Keep the 2-4 lean narrative tight sentences (not a list) that highlight causal links and transitions using connective wording (e.g., 'next', 'as a result', 'three days later'). "
    "Skip tangents unrelated to the dominant storyline; if you must acknowledge other threads, do so with a single closing sentence."
)

TIME_FORMATS: Tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H",
    "%Y-%m-%d %H",
    "%Y-%m-%d",
)


def has_temporal_info(text: str) -> bool:
    """
    检查文本是否包含时间信息
    
    识别的时间表达包括：
    - ISO日期: 2023-08-16
    - 完整日期: November 5, 2022 / in November 2023
    - 相对时间(锚定): the Friday before 2023-08-16 / two weeks before 2023-08-11
    - 时间状态: as of 2023-08-16 / after 2023-08-13
    - 相对时间(无锚定): last Friday / recently / yesterday
    """
    if not text:
        return False
    
    # ISO日期格式
    if re.search(r'\b\d{4}-\d{2}-\d{2}\b', text):
        return True
    
    # 月份名称
    months = r"January|February|March|April|May|June|July|August|September|October|November|December"
    
    # 完整日期格式
    if re.search(rf'\b(?:on|in|during|at)?\s*(?:{months})\s+\d{{1,2}},?\s+\d{{4}}\b', text, re.IGNORECASE):
        return True
    if re.search(rf'\b(?:in|during)\s+(?:{months})\s+\d{{4}}\b', text, re.IGNORECASE):
        return True
    
    # 相对时间表达(带锚定日期)
    time_units = r"day|week|month|year|weekend|semester"
    weekdays = r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
    
    if re.search(rf'\b(?:the\s+)?(?:{time_units})s?\s+(?:before|after)\s+\d{{4}}-\d{{2}}-\d{{2}}\b', text, re.IGNORECASE):
        return True
    if re.search(rf'\b(?:the\s+)?{weekdays}\s+(?:before|after)\s+\d{{4}}-\d{{2}}-\d{{2}}\b', text, re.IGNORECASE):
        return True
    if re.search(rf'\b\d+\s+(?:{time_units})s?\s+(?:before|after)\s+\d{{4}}-\d{{2}}-\d{{2}}\b', text, re.IGNORECASE):
        return True
    if re.search(rf'\bover\s+the\s+weekend\s+of\s+(?:{months})\s+\d{{1,2}},?\s+\d{{4}}\b', text, re.IGNORECASE):
        return True
    
    # 时间状态表达
    if re.search(r'\bas\s+of\s+\d{4}-\d{2}-\d{2}\b', text, re.IGNORECASE):
        return True
    if re.search(rf'\bas\s+of\s+(?:{months})\s+\d{{1,2}},?\s+\d{{4}}\b', text, re.IGNORECASE):
        return True
    if re.search(r'\b(?:after|before|since|until|by)\s+\d{4}-\d{2}-\d{2}\b', text, re.IGNORECASE):
        return True
    
    # 相对时间表达(无锚定)
    if re.search(rf'\blast\s+(?:{weekdays}|week|weekend|month|year|night)\b', text, re.IGNORECASE):
        return True
    if re.search(rf'\bnext\s+(?:{weekdays}|week|weekend|month|year)\b', text, re.IGNORECASE):
        return True
    if re.search(rf'\bthis\s+(?:{weekdays}|week|weekend|month|year|morning|afternoon|evening)\b', text, re.IGNORECASE):
        return True
    if re.search(r'\b(?:yesterday|today|tomorrow|recently|currently)\b', text, re.IGNORECASE):
        return True
    
    return False


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """解析时间戳字符串为datetime对象"""
    if value is None:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None

    try:
        return datetime.fromisoformat(candidate.rstrip("Z"))
    except ValueError:
        pass

    normalized = candidate.rstrip("Z")
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None


def format_time_with_weekday(value: str) -> str:
    """格式化时间字符串并附带星期信息"""
    if not value:
        return value

    parsed = parse_timestamp(value)
    if parsed is None:
        return value

    weekday = parsed.strftime("%A")
    return f"{value} ({weekday})"


def load_aggregations(path: Path) -> List[Dict[str, Any]]:
    """加载包含summary的aggregations"""
    aggregations: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            agg = json.loads(line)
            if "summary" in agg and agg["summary"]:
                aggregations.append(agg)
    return aggregations


def get_entry_type(entry: Dict[str, Any]) -> str:
    """获取entry的类型"""
    payload = entry.get("payload", {})
    entry_type = payload.get("entry_type") or entry.get("entry_type")
    if entry_type:
        return str(entry_type)
    memory_type = payload.get("memory_type") or entry.get("memory_type")
    if memory_type:
        return str(memory_type)
    return ""


def extract_entry_time(entry: Dict[str, Any]) -> str:
    """提取entry的时间戳"""
    payload = entry.get("payload", {})
    for key in ("mentioned_time", "time_stamp", "time", "timestamp"):
        value = payload.get(key)
        if value:
            return str(value)
    return ""


def extract_memory_text(entry: Dict[str, Any]) -> str:
    """提取entry的memory文本"""
    payload = entry.get("payload", {})
    for key in ("memory", "fact", "original_memory", "compressed_memory", "interaction"):
        value = payload.get(key)
        if value:
            return str(value)
    for key in ("memory", "fact", "original_memory", "compressed_memory", "interaction"):
        value = entry.get(key)
        if value:
            return str(value)
    return ""


def should_exclude_entry(
    entry: Dict[str, Any],
    bucket_time: datetime,
    bucket_entry_ids: Set[str],
    buffer_hours: float,
) -> bool:
    """判断entry是否应该被排除（基于时间缓冲区和bucket归属）"""
    entry_id = str(entry.get("id", ""))
    if entry_id in bucket_entry_ids:
        return True

    entry_time_str = extract_entry_time(entry)
    if not entry_time_str:
        return True

    entry_time = parse_timestamp(entry_time_str)
    if entry_time is None:
        return True

    time_diff = abs((entry_time - bucket_time).total_seconds()) / 3600
    return time_diff < buffer_hours

def extract_conversation_time(entry: Dict[str, Any], fallback: Optional[datetime] = None) -> str:
    """提取对话发生的时间戳"""
    payload = entry.get("payload", {})
    for key in ("conversation_time", "mention_time", "mentioned_time", "time_stamp", "timestamp", "time"):
        value = payload.get(key)
        if value:
            return str(value)
    for key in ("conversation_time", "mention_time", "mentioned_time", "time_stamp", "timestamp", "time"):
        value = entry.get(key)
        if value:
            return str(value)
    if fallback is not None:
        return fallback.isoformat()
    return ""




def sort_entries_by_time(entries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """按时间排序entries并返回索引映射"""
    timed_entries: List[Tuple[datetime, Dict[str, Any]]] = []
    for entry in entries:
        entry_time = parse_timestamp(extract_entry_time(entry))
        if entry_time is None:
            continue
        timed_entries.append((entry_time, entry))

    timed_entries.sort(key=lambda item: item[0])
    sorted_entries = [entry for _, entry in timed_entries]
    index_map = {str(entry.get("id", "")): idx for idx, entry in enumerate(sorted_entries)}
    return sorted_entries, index_map


def collect_context_entries(
    entry_id: str,
    sorted_entries: List[Dict[str, Any]],
    index_map: Dict[str, int],
    window: int,
    bucket_time: datetime,
) -> List[Dict[str, Any]]:
    """收集entry的时间上下文"""
    idx = index_map.get(entry_id)
    if idx is None:
        return []

    start = max(0, idx - window)
    end = min(len(sorted_entries), idx + window + 1)
    context: List[Dict[str, Any]] = []

    for pos in range(start, end):
        entry = sorted_entries[pos]
        payload = entry.get("payload", {})
        context.append(
            {
                "entry_id": str(entry.get("id", "")),
                "time": extract_entry_time(entry),
                "speaker": payload.get("speaker_name") or payload.get("speaker_id") or "",
                "memory": extract_memory_text(entry),
                "conversation_time": extract_conversation_time(entry, bucket_time),
                "is_target": pos == idx,
            }
        )

    return context


def retrieve_with_summary(
    summary: str,
    entries: List[Dict[str, Any]],
    bucket_time: datetime,
    bucket_entry_ids: Set[str],
    buffer_hours: float,
    retriever: VectorRetriever,
    search_limit: int,
    entry_lookup: Dict[str, Dict[str, Any]],
    temporal_only: bool = True,
) -> List[Dict[str, Any]]:
    """使用summary进行向量检索"""
    if not summary.strip():
        return []

    # 如果启用时间过滤，先筛选出包含时间信息的entries
    search_entries = entries
    if temporal_only:
        search_entries = [e for e in entries if has_temporal_info(extract_memory_text(e))]
        if len(search_entries) < len(entries):
            print(f"    时间过滤: {len(entries)} -> {len(search_entries)} entries")

    try:
        candidates = retriever.retrieve(search_entries, summary, limit=search_limit)
    except Exception as exc:
        print(f"检索失败: {exc}")
        return []

    filtered: List[Dict[str, Any]] = []
    for candidate in candidates:
        entry_id = candidate.get("id")
        if entry_id is None:
            continue
        original = entry_lookup.get(str(entry_id))
        if original is None:
            continue
        if should_exclude_entry(original, bucket_time, bucket_entry_ids, buffer_hours):
            continue
        filtered.append({"entry": original, "score": float(candidate.get("score", 0.0))})

    filtered.sort(key=lambda item: item["score"], reverse=True)
    return filtered[:RETRIEVAL_TOP_K]


def prepare_entry_snapshot(
    entry: Dict[str, Any],
    bucket_time: datetime,
    sorted_entries: List[Dict[str, Any]],
    index_map: Dict[str, int],
    window: int,
    score: Optional[float] = None,
) -> Dict[str, Any]:
    """准备entry的快照数据"""
    payload = entry.get("payload", {})
    entry_id = str(entry.get("id", ""))
    entry_time = extract_entry_time(entry)

    snapshot: Dict[str, Any] = {
        "entry_id": entry_id,
        "entry_type": get_entry_type(entry),
        "time": entry_time,
        "speaker": payload.get("speaker_name") or payload.get("speaker_id") or "",
        "memory": extract_memory_text(entry),
        "conversation_time": extract_conversation_time(entry, bucket_time),
    }

    if score is not None:
        snapshot["score"] = float(score)

    entry_time_dt = parse_timestamp(entry_time)
    if entry_time_dt is not None:
        time_distance = abs((entry_time_dt - bucket_time).total_seconds()) / 3600
        snapshot["time_distance_hours"] = time_distance

    snapshot["context"] = collect_context_entries(entry_id, sorted_entries, index_map, window, bucket_time)
    return snapshot


def run_summary_retrieval(
    qdrant_dir: Path,
    aggregations_path: Path,
    *,
    collections: Optional[Sequence[str]],
    openai_model: str,
    openai_api_key: str,
    openai_base_url: str,
    time_buffer_hours: float,
    search_limit: int,
    temporal_only: bool = True,
) -> List[Dict[str, Any]]:
    """执行summary检索的主流程"""
    if not qdrant_dir.exists():
        raise FileNotFoundError(f"Qdrant目录不存在: {qdrant_dir}")
    if not aggregations_path.exists():
        raise FileNotFoundError(f"Aggregations文件不存在: {aggregations_path}")

    print(f"加载aggregations...")
    aggregations = load_aggregations(aggregations_path)
    print(f"加载 {len(aggregations)} 个aggregations")

    if not aggregations:
        raise ValueError("没有找到包含summary的aggregation")

    collection_filter: Optional[Set[str]] = set(collections) if collections else None
    if collection_filter:
        aggregations = [agg for agg in aggregations if agg.get("collection") in collection_filter]
        print(f"过滤后剩余 {len(aggregations)} 个aggregations")

    if temporal_only:
        print("启用时间过滤：只检索包含时间信息的entries")

    cfg = BaseTextEmbedderConfig(
        model=openai_model,
        api_key=openai_api_key,
        openai_base_url=openai_base_url,
        embedding_dims=None,
    )
    embedder = TextEmbedderOpenAI(cfg)
    retriever = VectorRetriever(embedder)
    entry_loader = QdrantEntryLoader(str(qdrant_dir))

    buckets_by_collection: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for agg in aggregations:
        collection = agg.get("collection", "")
        if collection:
            buckets_by_collection[collection].append(agg)

    print(f"处理 {len(buckets_by_collection)} 个collections")

    results: List[Dict[str, Any]] = []

    for collection, collection_buckets in sorted(buckets_by_collection.items()):
        print(f"\n处理 Collection: {collection} ({len(collection_buckets)} buckets)")

        try:
            all_entries = entry_loader.load_entries(collection, with_vectors=True)
        except Exception as exc:
            print(f"加载collection失败: {exc}")
            continue

        if not all_entries:
            print("未找到entries，跳过")
            continue

        # 如果启用时间过滤，统计一下
        if temporal_only:
            temporal_count = sum(1 for e in all_entries if has_temporal_info(extract_memory_text(e)))
            print(f"  包含时间信息的entries: {temporal_count}/{len(all_entries)}")

        entry_lookup = {str(entry.get("id", "")): entry for entry in all_entries}
        sorted_entries, index_map = sort_entries_by_time(all_entries)
        if not sorted_entries:
            print("fact entries缺少时间信息，跳过")
            continue

        for bucket in sorted(collection_buckets, key=lambda b: b.get("bucket", "")):
            summary = (bucket.get("summary", "") or "").strip()
            if not summary:
                continue

            bucket_key = bucket.get("bucket", "")
            bucket_time = parse_timestamp(bucket_key)
            if bucket_time is None:
                print(f"无法解析bucket时间: {bucket_key}")
                continue
            
            bucket_entry_ids = {str(eid) for eid in bucket.get("entry_ids", [])}

            retrieved = retrieve_with_summary(
                summary,
                all_entries,
                bucket_time,
                bucket_entry_ids,
                time_buffer_hours,
                retriever,
                search_limit,
                entry_lookup,
                temporal_only=temporal_only,
            )

            retrieved_snapshots = [
                prepare_entry_snapshot(
                    item["entry"],
                    bucket_time,
                    sorted_entries,
                    index_map,
                    CONTEXT_WINDOW,
                    score=item.get("score"),
                )
                for item in retrieved
            ]

            summary_snapshots: List[Dict[str, Any]] = []
            missing_summary_entries: List[str] = []
            for entry_id in bucket_entry_ids:
                entry = entry_lookup.get(entry_id)
                if entry is None:
                    missing_summary_entries.append(entry_id)
                    continue
                summary_snapshots.append(
                    prepare_entry_snapshot(
                        entry,
                        bucket_time,
                        sorted_entries,
                        index_map,
                        CONTEXT_WINDOW,
                    )
                )

            bucket_result = {
                "collection": collection,
                "bucket": bucket_key,
                "bucket_time": bucket_time.isoformat(),
                "speakers": bucket.get("speakers", []),
                "summary": summary,
                "retrieved_entries": retrieved_snapshots,
                "summary_entries": summary_snapshots,
                "missing_summary_entry_ids": sorted(set(missing_summary_entries)),
                "aggregation": bucket,
            }

            results.append(bucket_result)

    print(f"\n生成 {len(results)} 个bucket结果")
    return results


def build_bucket_events(bucket: Dict[str, Any]) -> Dict[str, Any]:
    """构建bucket的事件数据"""
    def sort_by_time(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            entries,
            key=lambda e: parse_timestamp(e.get("time")) or datetime.max
        )

    summary_entries = sort_by_time(bucket.get("summary_entries") or [])
    retrieved_entries = sort_by_time(bucket.get("retrieved_entries") or [])

    speakers = set(bucket.get("speakers") or [])
    for entry in summary_entries + retrieved_entries:
        speaker = entry.get("speaker")
        if speaker:
            speakers.add(speaker)

    return {
        "collection": bucket.get("collection"),
        "bucket": bucket.get("bucket"),
        "bucket_time": bucket.get("bucket_time"),
        "summary": bucket.get("summary"),
        "summary_entries": summary_entries,
        "retrieved_entries": retrieved_entries,
        "speakers": sorted(speakers),
        "missing_summary_entry_ids": bucket.get("missing_summary_entry_ids") or [],
    }


def build_llm_messages(bucket_meta: Dict[str, Any]) -> List[Dict[str, str]]:
    """构建LLM的输入消息"""
    bucket_anchor = bucket_meta.get("bucket_time") or ""

    def render_entries(title: str, entries: Sequence[Dict[str, Any]]) -> List[str]:
        lines = [title]
        if not entries:
            lines.append("  (none)")
            return lines
        for idx, entry in enumerate(entries, start=1):
            event_time = entry.get("time") or ""
            conversation_time = entry.get("conversation_time") or bucket_anchor or ""
            speaker = entry.get("speaker") or ""
            tags: List[str] = []
            if conversation_time:
                tags.append(f"Conversation {format_time_with_weekday(conversation_time)}")
            if event_time and event_time != conversation_time:
                tags.append(f"Event {format_time_with_weekday(event_time)}")

            prefix = f"  {idx}. "
            if tags:
                prefix += f"[{ ' | '.join(tags) }]"
            prefix = prefix.rstrip()
            if speaker:
                prefix += f" {speaker}:"
            lines.append(prefix)
            lines.append(f"     {entry.get('memory', '')}")
        return lines

    lines: List[str] = []
    lines.extend(render_entries("Core Entries:", bucket_meta["summary_entries"]))
    lines.append("")
    lines.extend(render_entries("There are some Retrieved Entries that maybe related to our main entries:", bucket_meta["retrieved_entries"]))
    lines.append("")
    lines.append("Example responses (mirror the structure and connective wording):")
    lines.append("  On Saturday, 2024-03-02, Bob told Alice that he had filed the grant application, setting the plan in motion.")
    lines.append("  Three days later on Tuesday, 2024-03-05, Alice confirmed the reviewers wanted revisions, so Bob drafted updates that kept the timeline moving.")
    lines.append("  A week after 2024-03-12 (Tuesday), Bob reported the approval and scheduled the kickoff, showing the earlier edits paid off.")
    lines.append("  Yesterday at 2024-04-19 (Friday) referencing Monday's storm, Alice briefed Bob on the volunteer shift, and today she logged the first repair requests to keep momentum.")
    lines.append("  Two weeks before the launch on 2024-05-01 (Wednesday), Bob recruited testers; one week later on 2024-05-08 (Wednesday) Alice ran the dry run; on launch day 2024-05-15 (Wednesday) their prep absorbed the onboarding surge.")
    lines.append("")
    lines.append("Response requirements:")
    lines.append(TIMELINE_INSTRUCTION)

    user_prompt = "\n".join(lines)
    return [
        {
            "role": "system",
            "content": "You organize events strictly based on the provided entries.",
        },
        {"role": "user", "content": user_prompt},
    ]


def call_llm(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int = 120,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """调用LLM生成事件描述，返回文本、原始响应以及usage信息"""
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("LLM response did not contain choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        raise ValueError("LLM response did not contain text content")
    usage = data.get("usage") or {}
    return content, data, usage


def filter_buckets(
    results: Sequence[Dict[str, Any]],
    collections: Optional[Sequence[str]],
    buckets: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    """过滤buckets"""
    collection_set = set(collections) if collections else None
    bucket_set = set(buckets) if buckets else None

    filtered = [
        item
        for item in results
        if (not collection_set or item.get("collection") in collection_set)
        and (not bucket_set or item.get("bucket") in bucket_set)
    ]

    filtered.sort(
        key=lambda item: (
            str(item.get("collection", "")),
            str(item.get("bucket_time") or item.get("bucket") or ""),
        )
    )

    return filtered


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="构建事件时间线图谱")

    retrieval_group = parser.add_argument_group("检索配置")
    retrieval_group.add_argument("--qdrant-dir", type=Path, help="Qdrant数据目录")
    retrieval_group.add_argument("--aggregations", type=Path, help="包含summary的aggregations文件")
    retrieval_group.add_argument("--openai-model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding模型")
    retrieval_group.add_argument("--openai-api-key", default=DEFAULT_API_KEY, help="OpenAI API Key")
    retrieval_group.add_argument("--openai-base-url", default=DEFAULT_BASE_URL, help="OpenAI Base URL")
    retrieval_group.add_argument("--time-buffer-hours", type=float, default=1.5, 
                                help="时间缓冲区（小时）")
    retrieval_group.add_argument("--search-limit", type=int, default=50, help="检索候选数量")
    retrieval_group.add_argument("--retrieval-output", type=Path, help="保存检索结果的路径")
    retrieval_group.add_argument("--temporal-only", action="store_true", 
                                help="只检索包含时间信息的entries")

    parser.add_argument("--input", type=Path, help="已有的检索结果JSON文件")
    parser.add_argument("--collections", nargs="*", help="指定处理的collections")
    parser.add_argument("--buckets", nargs="*", help="指定处理的buckets")
    
    llm_group = parser.add_argument_group("LLM配置")
    llm_group.add_argument("--api-key", default=DEFAULT_API_KEY, help="LLM API Key")
    llm_group.add_argument("--api-base", default=DEFAULT_BASE_URL, help="LLM API Base URL")
    llm_group.add_argument("--model", default=DEFAULT_TIMELINE_MODEL, help="LLM模型名称")
    llm_group.add_argument("--temperature", type=float, default=0.0, help="生成温度")
    llm_group.add_argument("--max-tokens", type=int, default=800, help="最大token数")
    
    output_group = parser.add_argument_group("输出配置")
    output_group.add_argument("--events-output", type=Path, help="保存事件数据的路径")
    output_group.add_argument("--response-output", type=Path, help="保存LLM响应的路径")
    output_group.add_argument("--relations-output", type=Path, help="保存绑定时间序列关系的aggregations JSONL路径")

    args = parser.parse_args()

    has_qdrant = args.qdrant_dir is not None
    has_aggregations = args.aggregations is not None
    if has_qdrant != has_aggregations:
        parser.error("需要同时提供 --qdrant-dir 和 --aggregations")
    if not args.input and not has_qdrant:
        parser.error("必须提供 --input 或同时提供 --qdrant-dir 与 --aggregations")

    return args


def main() -> None:
    """主函数"""
    args = parse_args()

    # 第一步：执行检索或加载已有结果
    if args.qdrant_dir and args.aggregations:
        print("执行summary检索...")
        retrieval_results = run_summary_retrieval(
            args.qdrant_dir,
            args.aggregations,
            collections=args.collections,
            openai_model=args.openai_model,
            openai_api_key=args.openai_api_key,
            openai_base_url=args.openai_base_url,
            time_buffer_hours=args.time_buffer_hours,
            search_limit=args.search_limit,
            temporal_only=args.temporal_only,
        )
        
        if args.retrieval_output:
            output_path = args.retrieval_output.resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sanitized_results: List[Dict[str, Any]] = []
            for item in retrieval_results:
                if "aggregation" in item:
                    filtered_item = {k: v for k, v in item.items() if k != "aggregation"}
                    sanitized_results.append(filtered_item)
                else:
                    sanitized_results.append(item)
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(sanitized_results, fh, ensure_ascii=False, indent=2)
            print(f"检索结果已保存: {output_path}")
        
        results = retrieval_results
    else:
        print(f"加载已有结果: {args.input}")
        if not args.input.exists():
            raise FileNotFoundError(f"找不到输入文件: {args.input}")
        with args.input.open("r", encoding="utf-8") as fh:
            results = json.load(fh)

    # 第二步：过滤buckets
    filtered = filter_buckets(results, args.collections, args.buckets)
    print(f"过滤后得到 {len(filtered)} 个buckets")

    if not filtered:
        print("未找到匹配的bucket")
        return

    # 第三步：处理每个bucket，调用LLM生成事件描述
    print("\n开始生成事件描述...")
    timeline_payloads: List[Dict[str, Any]] = []
    responses: List[Dict[str, Any]] = []
    relation_records: List[Dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for idx, bucket in enumerate(filtered, start=1):
        bucket_meta = build_bucket_events(bucket)
        collection = bucket_meta.get("collection")
        bucket_id = bucket_meta.get("bucket")
        
        print(f"[{idx}/{len(filtered)}] 处理 {collection}::{bucket_id}")

        messages = build_llm_messages(bucket_meta)

        try:
            llm_text, llm_raw, llm_usage = call_llm(
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            print(f"  ✓ LLM生成完成")
            print(f"\n{'='*80}")
            print(f"LLM输出 [{collection}::{bucket_id}]:")
            print(f"{'='*80}")
            print(llm_text.strip())
            print()
            prompt_tokens = int(llm_usage.get("prompt_tokens") or 0)
            completion_tokens = int(llm_usage.get("completion_tokens") or 0)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_tokens += int(llm_usage.get("total_tokens") or prompt_tokens + completion_tokens)
        except Exception as exc:
            llm_text = f"[LLM调用失败] {exc}"
            llm_raw = {"error": str(exc)}
            llm_usage = {}
            print(f"  ✗ LLM调用失败: {exc}")

        responses.append(
            {
                "collection": collection,
                "bucket": bucket_id,
                "bucket_time": bucket_meta.get("bucket_time"),
                "model": args.model,
                "prompt_messages": messages,
                "response": llm_text,
                "raw": llm_raw,
                "usage": llm_usage,
            }
        )

        timeline_payloads.append(bucket_meta)

        agg_source = bucket.get("aggregation")
        if agg_source is not None:
            relation_entry = deepcopy(agg_source)
        else:
            relation_entry = {
                "collection": collection,
                "bucket": bucket_id,
                "summary": bucket.get("summary"),
                "entry_ids": bucket.get("entry_ids"),
            }
        relation_entry["timeline_relation"] = {
            "timeline_text": llm_text,
            "timeline_payload": bucket_meta,
            "usage": llm_usage,
        }
        relation_records.append(relation_entry)
        bucket.pop("aggregation", None)

    # 第四步：保存结果
    if args.events_output:
        output_path = args.events_output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(timeline_payloads, fh, ensure_ascii=False, indent=2)
        print(f"\n事件数据已保存: {output_path}")

    if args.response_output:
        response_path = args.response_output.resolve()
        response_path.parent.mkdir(parents=True, exist_ok=True)
        with response_path.open("w", encoding="utf-8") as fh:
            json.dump(responses, fh, ensure_ascii=False, indent=2)
        print(f"LLM响应已保存: {response_path}")

    if args.relations_output:
        relations_path = args.relations_output.resolve()
        relations_path.parent.mkdir(parents=True, exist_ok=True)
        with relations_path.open("w", encoding="utf-8") as fh:
            for record in relation_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"时间序列关系已保存: {relations_path}")

    if total_tokens:
        print(
            f"\nToken usage summary — prompt: {total_prompt_tokens}, completion: {total_completion_tokens}, total: {total_tokens}"
        )

    print(f"\n完成！共处理 {len(filtered)} 个buckets")


if __name__ == "__main__":
    main()