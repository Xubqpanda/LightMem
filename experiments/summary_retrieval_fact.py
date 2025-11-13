import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists():
    import sys
    sys.path.append(str(SRC_PATH))

from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig  # type: ignore[import]
from lightmem.factory.text_embedder.openai import TextEmbedderOpenAI  # type: ignore[import]
from retrievers import QdrantEntryLoader, VectorRetriever  # type: ignore[import]

DEFAULT_API_KEY = 'sk-elh8U89D8GUNipT9CdDaF39a1e5d44649e48B392E98cC24d'
DEFAULT_BASE_URL = 'https://api.gpts.vin/v1'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用summary检索相关entry并查看上下文")
    parser.add_argument("--qdrant-dir", type=Path, required=True, help="Qdrant数据目录")
    parser.add_argument("--aggregations", type=Path, required=True, help="包含summary的hourly_aggregations.jsonl文件")
    parser.add_argument("--collections", nargs="*", help="要分析的collections")
    parser.add_argument("--top-k", type=int, default=5, help="检索top-k个相关entry")
    parser.add_argument("--openai-model", default="text-embedding-3-small", help="OpenAI embedding模型")
    parser.add_argument("--openai-api-key", default=DEFAULT_API_KEY, help="OpenAI API Key")
    parser.add_argument("--openai-base-url", default=DEFAULT_BASE_URL, help="OpenAI Base URL")
    parser.add_argument("--time-buffer-hours", type=float, default=1.5, 
                       help="时间缓冲区：排除summary所在小时前后各N小时的entry（默认1.5小时）")
    parser.add_argument("--search-limit", type=int, default=50,
                       help="检索的初始候选数量（过滤后取top-k，默认50）")
    parser.add_argument("--context-window", type=int, default=3,
                        help="上下文窗口大小，前后各包含的entry数量")
    parser.add_argument("--debug-entry-types", action="store_true",
                        help="输出详细的entry_type调试信息")
    return parser.parse_args()


def load_aggregations(path: Path) -> List[Dict]:
    """加载已生成的aggregations（包含summary）"""
    aggregations = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            agg = json.loads(line)
            if "summary" in agg and agg["summary"]:
                aggregations.append(agg)
    return aggregations


def parse_bucket_time(bucket_str: str) -> datetime:
    """解析bucket时间字符串为datetime对象"""
    try:
        return datetime.fromisoformat(bucket_str.rstrip("Z"))
    except Exception:
        return datetime.strptime(bucket_str[:13], "%Y-%m-%dT%H")


def extract_entry_time(entry: Dict[str, Any]) -> str:
    """从entry中提取时间字符串"""
    payload = entry.get("payload", {})
    for key in ("mentioned_time", "time_stamp", "time", "timestamp"):
        value = payload.get(key)
        if value:
            return str(value)
    return ""


def parse_time_string(time_str: str) -> Optional[datetime]:
    if not time_str:
        return None
    try:
        return datetime.fromisoformat(time_str.rstrip("Z"))
    except ValueError:
        try:
            return datetime.strptime(time_str[:19], "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return None


def should_exclude_entry(entry: Dict[str, Any], bucket_time: datetime, bucket_entry_ids: set, buffer_hours: float) -> bool:
    """
    判断entry是否应该被排除
    1. 是bucket自己的entry
    2. 在时间缓冲区内
    """
    entry_id = str(entry.get('id', ''))
    
    # 排除bucket自己的entry
    if entry_id in bucket_entry_ids:
        return True
    
    # 排除时间缓冲区内的entry
    entry_time_str = extract_entry_time(entry)

    if not entry_time_str:
        return True
    
    entry_time = parse_time_string(entry_time_str)
    if entry_time is None:
        return True

    time_diff = abs((entry_time - bucket_time).total_seconds() / 3600)
    return time_diff < buffer_hours


def compute_time_distance_hours(reference_time: datetime, entry_time_str: str) -> Optional[float]:
    entry_time = parse_time_string(entry_time_str)
    if entry_time is None:
        return None
    return abs((entry_time - reference_time).total_seconds()) / 3600


def sort_entries_by_time(entries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    timed_entries: List[Tuple[datetime, Dict[str, Any]]] = []
    for entry in entries:
        entry_time = parse_time_string(extract_entry_time(entry))
        if entry_time is None:
            continue
        timed_entries.append((entry_time, entry))

    timed_entries.sort(key=lambda item: item[0])
    sorted_entries = [entry for _, entry in timed_entries]
    index_map = {str(entry.get('id', '')): idx for idx, entry in enumerate(sorted_entries)}
    return sorted_entries, index_map


def collect_context_entries(entry_id: str, sorted_entries: List[Dict[str, Any]], index_map: Dict[str, int], window: int) -> List[Dict[str, Any]]:
    idx = index_map.get(entry_id)
    if idx is None:
        return []

    start = max(0, idx - window)
    end = min(len(sorted_entries), idx + window + 1)
    context: List[Dict[str, Any]] = []

    for pos in range(start, end):
        entry = sorted_entries[pos]
        payload = entry.get('payload', {})
        context.append({
            "entry_id": str(entry.get('id', '')),
            "time": extract_entry_time(entry),
            "speaker": payload.get('speaker_name') or payload.get('speaker_id') or '',
            "memory": payload.get('memory') or payload.get('original_memory') or payload.get('compressed_memory') or '',
            "is_target": pos == idx,
        })

    return context


def retrieve_with_summary(
    summary: str,
    all_entries: List[Dict[str, Any]],
    bucket_time: datetime,
    bucket_entry_ids: set,
    buffer_hours: float,
    retriever: VectorRetriever,
    top_k: int,
    search_limit: int,
    entry_lookup: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """使用summary检索相关entry，并过滤后返回原始entry与得分"""

    if not summary.strip():
        return []

    try:
        candidates = retriever.retrieve(all_entries, summary, limit=search_limit)
    except Exception as exc:
        print(f"Warning: retrieval failed: {exc}")
        return []

    filtered: List[Dict[str, Any]] = []
    for candidate in candidates:
        entry_id = candidate.get('id')
        if entry_id is None:
            continue
        original = entry_lookup.get(str(entry_id))
        if original is None:
            continue
        if should_exclude_entry(original, bucket_time, bucket_entry_ids, buffer_hours):
            continue
        filtered.append({
            "entry": original,
            "score": float(candidate.get('score', 0.0)),
        })

    filtered.sort(key=lambda item: item['score'], reverse=True)
    return filtered[:top_k]


def get_entry_type(entry: Dict[str, Any]) -> str:
    payload = entry.get("payload", {})
    entry_type = payload.get("entry_type") or entry.get("entry_type")
    if entry_type:
        return str(entry_type)
    memory_type = payload.get("memory_type") or entry.get("memory_type")
    if memory_type:
        return str(memory_type)
    return ""


def is_fact_entry(entry: Dict[str, Any]) -> bool:
    return get_entry_type(entry).lower() == "fact"


def extract_memory_text(entry: Dict[str, Any]) -> str:
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


def build_entry_record(
    entry: Dict[str, Any],
    source: str,
    bucket_time: datetime,
    sorted_entries: List[Dict[str, Any]],
    index_map: Dict[str, int],
    window: int,
    score: Optional[float] = None,
) -> Dict[str, Any]:
    payload = entry.get("payload", {})
    entry_id = str(entry.get("id", ""))
    entry_time = extract_entry_time(entry)
    record: Dict[str, Any] = {
        "entry_id": entry_id,
        "sources": [source],
        "entry_type": get_entry_type(entry),
        "time": entry_time,
        "speaker": payload.get("speaker_name") or payload.get("speaker_id") or "",
        "memory": extract_memory_text(entry),
    }
    if score is not None:
        record["score"] = float(score)
    time_distance = compute_time_distance_hours(bucket_time, entry_time)
    if time_distance is not None:
        record["time_distance_hours"] = time_distance
    record["context"] = collect_context_entries(entry_id, sorted_entries, index_map, window)
    return record


def merge_entry_record(record: Dict[str, Any], container: Dict[str, Dict[str, Any]]):
    entry_id = record.get("entry_id", "")
    if not entry_id:
        return

    existing = container.get(entry_id)
    if existing is None:
        container[entry_id] = record
        return

    existing_sources = existing.setdefault("sources", [])
    sources_seen = set(existing_sources)
    for src in record.get("sources", []):
        if src not in sources_seen:
            existing_sources.append(src)
            sources_seen.add(src)

    if "score" in record:
        existing["score"] = record["score"]
    if record.get("time_distance_hours") is not None:
        existing["time_distance_hours"] = record["time_distance_hours"]

    if not existing.get("memory") and record.get("memory"):
        existing["memory"] = record["memory"]

    if not existing.get("entry_type") and record.get("entry_type"):
        existing["entry_type"] = record["entry_type"]

    if not existing.get("time") and record.get("time"):
        existing["time"] = record["time"]

    if not existing.get("speaker") and record.get("speaker"):
        existing["speaker"] = record["speaker"]

    existing_context = existing.get("context", [])
    context_keys = {
        (ctx.get("entry_id"), ctx.get("is_target"), ctx.get("time"))
        for ctx in existing_context
    }
    for ctx in record.get("context", []):
        key = (ctx.get("entry_id"), ctx.get("is_target"), ctx.get("time"))
        if key not in context_keys:
            existing_context.append(ctx)
            context_keys.add(key)

    def _context_sort_key(ctx: Dict[str, Any]) -> datetime:
        dt = parse_time_string(str(ctx.get("time", "")))
        return dt if dt is not None else datetime.min

    existing_context.sort(key=_context_sort_key)
    existing["context"] = existing_context


def sort_records_by_time(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _sort_key(rec: Dict[str, Any]) -> datetime:
        dt = parse_time_string(rec.get("time", ""))
        return dt if dt is not None else datetime.min

    return sorted(records, key=_sort_key)


def print_entry_type_debug(entry: Dict[str, Any], prefix: str = ""):
    """打印单个entry的详细类型信息（用于调试）"""
    payload = entry.get("payload", {})
    entry_id = str(entry.get('id', ''))[:50]
    
    print(f"{prefix}Entry ID: {entry_id}")
    print(f"{prefix}  payload.entry_type: {payload.get('entry_type', '(未设置)')}")
    print(f"{prefix}  entry.entry_type: {entry.get('entry_type', '(未设置)')}")
    print(f"{prefix}  payload.memory_type: {payload.get('memory_type', '(未设置)')}")
    print(f"{prefix}  entry.memory_type: {entry.get('memory_type', '(未设置)')}")
    print(f"{prefix}  get_entry_type() 返回: '{get_entry_type(entry)}'")
    print(f"{prefix}  is_fact_entry() 返回: {is_fact_entry(entry)}")
    print(f"{prefix}  payload 所有键: {list(payload.keys())}")


def main():
    args = parse_args()
    
    # 验证文件存在
    if not args.qdrant_dir.exists():
        raise FileNotFoundError(f"Qdrant目录不存在: {args.qdrant_dir}")
    if not args.aggregations.exists():
        raise FileNotFoundError(f"Aggregations文件不存在: {args.aggregations}")
    
    print(f"加载aggregations (包含summary)...")
    aggregations = load_aggregations(args.aggregations)
    print(f"共加载 {len(aggregations)} 个包含summary的aggregations")
    
    if len(aggregations) == 0:
        print("错误: 没有找到包含summary的aggregation！")
        return
    
    # 过滤collections
    if args.collections:
        aggregations = [agg for agg in aggregations if agg.get("collection") in args.collections]
        print(f"过滤后剩余 {len(aggregations)} 个aggregations")
    
    # 初始化组件
    cfg = BaseTextEmbedderConfig(
        model=args.openai_model,
        api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        embedding_dims=None,
    )
    embedder = TextEmbedderOpenAI(cfg)
    retriever = VectorRetriever(embedder)
    entry_loader = QdrantEntryLoader(str(args.qdrant_dir))
    
    # 按collection分组aggregations（优化：同一个collection的buckets一起处理）
    buckets_by_collection = defaultdict(list)
    for agg in aggregations:
        collection = agg.get("collection", "")
        if collection:
            buckets_by_collection[collection].append(agg)
    
    print(f"准备处理 {len(buckets_by_collection)} 个collections")

    for collection, collection_buckets in sorted(buckets_by_collection.items()):
        print("\n" + "=" * 80)
        print(f"Collection: {collection} ({len(collection_buckets)} buckets)")
        print("=" * 80)

        try:
            all_entries = entry_loader.load_entries(collection, with_vectors=True)
        except Exception as exc:
            print(f"加载 collection {collection} 失败: {exc}")
            continue

        if not all_entries:
            print("未找到任何entries，跳过该collection")
            continue

        # ===== 添加调试信息：统计entry_type分布 =====
        entry_type_counter = Counter()
        entry_type_samples = {}
        
        for entry in all_entries:
            etype = get_entry_type(entry)
            etype_key = etype if etype else "(空字符串)"
            entry_type_counter[etype_key] += 1
            
            # 保存每种类型的第一个样例
            if etype_key not in entry_type_samples:
                entry_type_samples[etype_key] = entry
        
        print(f"\n{'='*60}")
        print(f"[调试] Collection '{collection}' 的 entry_type 统计信息")
        print(f"{'='*60}")
        print(f"总 entries 数量: {len(all_entries)}")
        print(f"\nentry_type 分布:")
        
        for etype, count in entry_type_counter.most_common():
            percentage = (count / len(all_entries)) * 100
            print(f"  - '{etype}': {count} 个 ({percentage:.1f}%)")
        
        # 统计通过 is_fact_entry() 的数量
        fact_entries = [e for e in all_entries if is_fact_entry(e)]
        print(f"\n通过 is_fact_entry() 过滤的: {len(fact_entries)} 个 ({len(fact_entries)/len(all_entries)*100:.1f}%)")
        
        # 如果开启详细调试，显示每种类型的样例
        if args.debug_entry_types:
            print(f"\n{'='*60}")
            print("每种 entry_type 的详细样例:")
            print(f"{'='*60}")
            for etype in sorted(entry_type_counter.keys()):
                print(f"\n类型 '{etype}' 的样例:")
                sample_entry = entry_type_samples[etype]
                print_entry_type_debug(sample_entry, prefix="  ")
        
        print(f"{'='*60}\n")
        # ===== 调试信息结束 =====

        entry_lookup = {str(entry.get('id', '')): entry for entry in all_entries}
        sorted_entries, index_map = sort_entries_by_time(all_entries)

        if not sorted_entries:
            print("该collection的entries缺少时间信息，无法生成上下文")
            continue

        for bucket in sorted(collection_buckets, key=lambda b: b.get("bucket", "")):
            summary = (bucket.get("summary", "") or "").strip()
            if not summary:
                print(f"Bucket {bucket.get('bucket', 'unknown')} 无summary，跳过")
                continue

            bucket_key = bucket.get("bucket", "")
            bucket_time = parse_bucket_time(bucket_key)
            bucket_entry_ids = {str(eid) for eid in bucket.get("entry_ids", [])}

            retrieved = retrieve_with_summary(
                summary,
                all_entries,
                bucket_time,
                bucket_entry_ids,
                args.time_buffer_hours,
                retriever,
                args.top_k,
                args.search_limit,
                entry_lookup,
            )

            # ===== 添加检索过程的调试信息 =====
            print(f"\n[调试] Bucket {bucket_key} 检索统计:")
            print(f"  检索返回候选: {len(retrieved)} 个")
            
            retrieved_fact_count = sum(1 for item in retrieved if is_fact_entry(item.get("entry")))
            retrieved_non_fact_count = len(retrieved) - retrieved_fact_count
            print(f"  其中 fact 类型: {retrieved_fact_count} 个")
            print(f"  其中非 fact 类型: {retrieved_non_fact_count} 个")
            
            if retrieved_non_fact_count > 0 and args.debug_entry_types:
                print(f"  被过滤掉的非 fact 类型样例:")
                for item in retrieved[:3]:  # 显示前3个
                    entry = item.get("entry")
                    if entry and not is_fact_entry(entry):
                        etype = get_entry_type(entry)
                        eid = str(entry.get('id', ''))[:50]
                        print(f"    - ID {eid}, type='{etype}'")
            # ===== 调试信息结束 =====

            combined_records: Dict[str, Dict[str, Any]] = {}
            retrieved_fact_ids: List[str] = []

            for item in retrieved:
                entry = item.get("entry")
                if entry is None or not is_fact_entry(entry):
                    continue
                record = build_entry_record(
                    entry,
                    source="retrieved",
                    bucket_time=bucket_time,
                    sorted_entries=sorted_entries,
                    index_map=index_map,
                    window=args.context_window,
                    score=item.get("score"),
                )
                merge_entry_record(record, combined_records)
                retrieved_fact_ids.append(record["entry_id"])

            summary_fact_ids: List[str] = []
            missing_summary_entries: List[str] = []
            
            # ===== 添加 bucket summary entries 的调试信息 =====
            bucket_fact_count = 0
            bucket_non_fact_count = 0
            
            for entry_id in bucket_entry_ids:
                entry = entry_lookup.get(entry_id)
                if entry is None:
                    missing_summary_entries.append(entry_id)
                    continue
                
                if is_fact_entry(entry):
                    bucket_fact_count += 1
                    record = build_entry_record(
                        entry,
                        source="bucket_summary",
                        bucket_time=bucket_time,
                        sorted_entries=sorted_entries,
                        index_map=index_map,
                        window=args.context_window,
                    )
                    merge_entry_record(record, combined_records)
                    summary_fact_ids.append(record["entry_id"])
                else:
                    bucket_non_fact_count += 1
            
            print(f"  Bucket 自身 entries: {len(bucket_entry_ids)} 个")
            print(f"    其中 fact 类型: {bucket_fact_count} 个")
            print(f"    其中非 fact 类型: {bucket_non_fact_count} 个")
            print(f"    缺失的 entries: {len(missing_summary_entries)} 个")
            # ===== 调试信息结束 =====

            retrieved_fact_ids = list(dict.fromkeys(retrieved_fact_ids))
            summary_fact_ids = list(dict.fromkeys(summary_fact_ids))
            missing_summary_entries = sorted(set(missing_summary_entries))

            merged_records = sort_records_by_time(list(combined_records.values()))

            payload = {
                "collection": collection,
                "bucket": bucket_key,
                "bucket_time": bucket_time.isoformat(),
                "speakers": bucket.get("speakers", []),
                "summary": summary,
                "retrieval_settings": {
                    "top_k": args.top_k,
                    "search_limit": args.search_limit,
                    "context_window": args.context_window,
                    "time_buffer_hours": args.time_buffer_hours,
                },
                "stats": {
                    "retrieved_total": len(retrieved),
                    "retrieved_fact_count": len(retrieved_fact_ids),
                    "summary_fact_count": len(summary_fact_ids),
                    "merged_count": len(merged_records),
                },
                "retrieved_fact_entry_ids": retrieved_fact_ids,
                "summary_fact_entry_ids": summary_fact_ids,
                "missing_summary_entry_ids": missing_summary_entries,
                "merged_fact_entries": merged_records,
            }

            print("\n" + "-" * 80)
            print(f"输出结果: {collection}::{bucket_key}")
            print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()