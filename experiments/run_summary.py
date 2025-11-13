import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists():
    import sys
    sys.path.append(str(SRC_PATH))

from lightmem.memory.utils import MemoryEntry  # type: ignore[import]
from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig  # type: ignore[import]
from lightmem.factory.text_embedder.openai import TextEmbedderOpenAI  # type: ignore[import]
from retrievers import QdrantEntryLoader  # type: ignore[import]

# LLM API 配置
LLM_API_KEY = 'sk-elh8U89D8GUNipT9CdDaF39a1e5d44649e48B392E98cC24d'
LLM_API_BASE_URL = 'https://api.gpts.vin/v1'
LLM_MODEL = 'gpt-4o-mini'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Qdrant entries, embed them, and generate summaries")
    parser.add_argument("--qdrant-dir", type=Path, required=True, help="Path to Qdrant data directory")
    parser.add_argument(
        "--collections",
        nargs="*",
        help="Collections to process (default: all collections listed in meta.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hourly_aggregations.jsonl"),
        help="Output JSONL file that will store aggregated text, vectors, and summaries",
    )
    parser.add_argument("--openai-model", default="text-embedding-3-small", help="OpenAI embedding model")
    parser.add_argument("--openai-api-key", default=LLM_API_KEY, help="Embedding API key (default: same as LLM)")
    parser.add_argument("--openai-base-url", default=LLM_API_BASE_URL, help="Embedding API base URL (default: same as LLM)")
    parser.add_argument(
        "--min-entries-per-bucket",
        type=int,
        default=1,
        help="Skip aggregated buckets smaller than this size",
    )
    parser.add_argument(
        "--target-hour",
        type=str,
        default=None,
        help=(
            "Target hour to process in format YYYY-MM-DDTHH (e.g. 2025-11-10T14). "
            "If not provided, all hours will be processed."
        ),
    )
    parser.add_argument("--print-buckets", action="store_true", help="Print aggregated text for processed buckets")
    parser.add_argument("--skip-summary", action="store_true", help="Skip summary generation")
    return parser.parse_args()


def load_collections(qdrant_dir: Path, collections: Optional[List[str]]) -> List[str]:
    if collections:
        return collections
    meta_path = qdrant_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("meta.json missing and no collections specified")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return sorted(meta.get("collections", {}).keys())


def load_entries_for_collection(loader: QdrantEntryLoader, collection: str) -> Tuple[List[Dict], List[MemoryEntry]]:
    points = loader.load_entries(collection, with_vectors=False)
    if not points:
        return [], []
    entries: List[MemoryEntry] = []
    for pt in points:
        payload: Dict = pt.get("payload", {})
        memory = payload.get("memory", "")
        time_stamp = payload.get("mentioned_time") or payload.get("time_stamp") or ""
        entry = MemoryEntry(
            id=str(pt.get("id")),
            time_stamp=time_stamp,
            memory=memory,
            topic_id=payload.get("topic_id"),
            speaker_id=payload.get("speaker_id", ""),
            speaker_name=payload.get("speaker_name", ""),
        )
        entries.append(entry)
    return points, entries


def bucket_key(time_stamp: str) -> str:
    if not time_stamp:
        return "unknown"
    cleaned = time_stamp.rstrip("Z")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        try:
            dt = datetime.strptime(cleaned.split(".")[0], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            # fallback: return first 13 chars (YYYY-MM-DDTHH)
            return time_stamp[:13]
    dt = dt.replace(minute=0, second=0, microsecond=0)
    return dt.isoformat()


def aggregate_entries(entries: List[MemoryEntry], min_size: int) -> Dict[str, List[MemoryEntry]]:
    buckets: Dict[str, List[MemoryEntry]] = defaultdict(list)
    for entry in entries:
        key = bucket_key(entry.time_stamp)
        buckets[key].append(entry)
    filtered: Dict[str, List[MemoryEntry]] = {}
    for key, items in buckets.items():
        if len(items) < min_size:
            continue
        items.sort(key=lambda e: e.time_stamp or "")
        filtered[key] = items
    return filtered


def build_openai_embedder(model: str, api_key: Optional[str], base_url: Optional[str]) -> TextEmbedderOpenAI:
    cfg = BaseTextEmbedderConfig(
        model=model,
        api_key=api_key,
        openai_base_url=base_url,
        embedding_dims=None,
    )
    return TextEmbedderOpenAI(cfg)


def to_aggregated_text(entries: Iterable[MemoryEntry]) -> str:
    lines: List[str] = []
    for entry in entries:
        speaker = entry.speaker_name or entry.speaker_id or "?"
        timestamp = entry.time_stamp or ""
        lines.append(f"[{timestamp}] {speaker}: {entry.memory}")
    return "\n".join(lines)


def create_summary_prompt(aggregated_text: str, bucket: str, speakers: List[str]) -> str:
    """创建用于生成 summary 的 prompt"""
    
    prompt = f"""You are a professional conversation summarization assistant. 

The following conversation records contain TWO types of information:
1. **Factual information**: concrete events, plans, opinions, preferences
2. **Interaction patterns**: how speakers relate to, support, and respond to each other

Both types are important and should be preserved in the summary.

Conversation Time: {bucket}
Participants: {', '.join(speakers)}

Conversation Records:
{aggregated_text}

Please generate a summary with the following requirements:

CRITICAL - What to PRESERVE:
- Specific concrete details: dates, times, locations, names of things
- Key emotional transitions and psychological changes 
- Concrete action plans
- Important quotes or specific expressions when they capture essential meaning

What to DO:
1. Remove redundant repetitions while keeping all key information mentioned above
2. Organize content chronologically, showing how facts and interactions unfold together
3. Highlight causal relationships (e.g., "X happened, which gave Y the courage to do Z")
4. Balance factual timeline with emotional/relational dynamics
5. Use fluent, concise natural language
6. Keep length between 150-300 words

Output the summary directly without any additional explanations or format markers."""
    
    return prompt


def _normalize_usage(usage_obj: object) -> Dict[str, int]:
    """提取并标准化 OpenAI 返回的 token usage 信息"""
    usage: Dict[str, int] = {}
    if usage_obj is None:
        return usage

    def _fetch(keys: Tuple[str, ...]) -> Optional[int]:
        for key in keys:
            value: Optional[int]
            if isinstance(usage_obj, dict):
                raw = usage_obj.get(key)  # type: ignore[index]
            else:
                raw = getattr(usage_obj, key, None)
            if raw is not None:
                try:
                    value = int(raw)
                except (TypeError, ValueError):
                    continue
                return value
        return None

    mappings = {
        "prompt_tokens": ("prompt_tokens", "input_tokens"),
        "completion_tokens": ("completion_tokens", "output_tokens"),
        "total_tokens": ("total_tokens",),
    }

    for target, keys in mappings.items():
        value = _fetch(keys)
        if value is not None:
            usage[target] = value

    if "total_tokens" not in usage:
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if prompt is not None and completion is not None:
            usage["total_tokens"] = prompt + completion

    return usage


def generate_summary(client: OpenAI, aggregated_text: str, bucket: str, speakers: List[str]) -> Tuple[str, Dict[str, int]]:
    """调用 GPT API 生成 summary 并返回 token usage"""
    prompt = create_summary_prompt(aggregated_text, bucket, speakers)
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional conversation summarization assistant who excels at extracting key information from lengthy conversation records and generating concise, coherent summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        summary = response.choices[0].message.content.strip()
        usage = _normalize_usage(getattr(response, "usage", None))
        return summary, usage
        
    except Exception as e:
        print(f"生成 summary 时出错: {e}")
        return "", {}


def save_aggregations(
    output_path: Path,
    collection: str,
    bucket_map: Dict[str, List[MemoryEntry]],
    vectors: List[List[float]],
    bucket_keys: List[str],
    llm_client: Optional[OpenAI],
    skip_summary: bool,
) -> Tuple[int, int, Dict[str, int]]:
    """保存聚合结果，包括summary（如果启用），并统计 token usage"""
    success_count = 0
    error_count = 0
    usage_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    
    with output_path.open("a", encoding="utf-8") as fout:
        for idx, (key, vector) in enumerate(zip(bucket_keys, vectors)):
            entries = bucket_map[key]
            speakers = sorted({(e.speaker_name or e.speaker_id or "?") for e in entries})
            aggregated_text = to_aggregated_text(entries)
            start_time = entries[0].time_stamp
            end_time = entries[-1].time_stamp
            
            record = {
                "collection": collection,
                "bucket": key,
                "entry_ids": [e.id for e in entries],
                "start_time": start_time,
                "end_time": end_time,
                "speakers": speakers,
                "aggregated_text": aggregated_text,
                "vector": vector,
            }
            
            # 生成summary（如果需要）
            if not skip_summary and llm_client:
                summary, usage = generate_summary(llm_client, aggregated_text, key, speakers)
                if summary:
                    record["summary"] = summary
                    if usage:
                        record["summary_usage"] = usage
                        for token_key, value in usage.items():
                            if token_key in usage_totals:
                                usage_totals[token_key] += value
                    success_count += 1
                else:
                    record["summary"] = ""
                    error_count += 1
            
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    return success_count, error_count, usage_totals


def main() -> None:
    args = parse_args()

    if not args.qdrant_dir.exists():
        raise FileNotFoundError(f"Qdrant directory not found: {args.qdrant_dir}")

    collections = load_collections(args.qdrant_dir, args.collections)
    loader = QdrantEntryLoader(str(args.qdrant_dir))
    embedder = build_openai_embedder(
        model=args.openai_model,
        api_key=args.openai_api_key,
        base_url=args.openai_base_url,
    )
    
    # 初始化LLM客户端（如果需要生成summary）
    llm_client = None
    if not args.skip_summary:
        llm_client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_API_BASE_URL
        )
        print(f"LLM模型: {LLM_MODEL}")

    if args.output.exists():
        args.output.unlink()

    total_success = 0
    total_error = 0
    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    for coll in collections:
        _, entries = load_entries_for_collection(loader, coll)
        if not entries:
            print(f"[{coll}] no entries found")
            continue
        bucket_map = aggregate_entries(entries, args.min_entries_per_bucket)
        if not bucket_map:
            print(f"[{coll}] no buckets after filtering")
            continue

        # if target_hour specified, only keep that bucket key
        if args.target_hour:
            # Normalize input target hour to match our bucket format (ISO hour)
            try:
                parsed = datetime.fromisoformat(args.target_hour)
                target_key = parsed.replace(minute=0, second=0, microsecond=0).isoformat()
            except ValueError:
                # allow inputs like YYYY-MM-DDTHH by padding
                target_key = args.target_hour
            if target_key not in bucket_map:
                print(f"[{coll}] target hour {args.target_hour} not present in collection")
                continue
            process_keys = [target_key]
        else:
            process_keys = sorted(bucket_map.keys())

        # 生成embeddings
        aggregated_texts = [to_aggregated_text(bucket_map[k]) for k in process_keys]
        vectors = embedder.embed(aggregated_texts)
        
        # 保存结果并生成summaries
        if not args.skip_summary:
            print(f"[{coll}] Generating summaries...")
            success, error, usage_totals = save_aggregations(
                args.output, coll, bucket_map, vectors, process_keys, llm_client, args.skip_summary
            )
            total_success += success
            total_error += error
            for key in total_usage:
                total_usage[key] += usage_totals.get(key, 0)
        else:
            _ = save_aggregations(
                args.output, coll, bucket_map, vectors, process_keys, None, args.skip_summary
            )

        sizes = [len(bucket_map[key]) for key in process_keys]
        print(f"[{coll}] processed_buckets={len(process_keys)} | avg_size={np.mean(sizes):.2f} | max_size={max(sizes)}")
        if not args.skip_summary:
            print(f"[{coll}] summaries: success={success}, error={error}")
            print(
                f"[{coll}] token_usage: prompt={usage_totals.get('prompt_tokens', 0)} | "
                f"completion={usage_totals.get('completion_tokens', 0)} | total={usage_totals.get('total_tokens', 0)}"
            )

        if args.print_buckets:
            for k in process_keys:
                print(f"--- {coll} | bucket={k} | size={len(bucket_map[k])} ---")
                print(bucket_map[k][0].memory if bucket_map[k] else "")

    if not args.skip_summary:
        print(f"\n总计 - 成功生成summary: {total_success}, 失败: {total_error}")
        print(
            "总计 - Token usage: "
            f"prompt={total_usage.get('prompt_tokens', 0)} | "
            f"completion={total_usage.get('completion_tokens', 0)} | "
            f"total={total_usage.get('total_tokens', 0)}"
        )
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()