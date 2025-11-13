import argparse
import json
from pathlib import Path

from build_graph import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    run_summary_retrieval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用summary检索相关entry并查看上下文")
    parser.add_argument("--qdrant-dir", type=Path, required=True, help="Qdrant数据目录")
    parser.add_argument("--aggregations", type=Path, required=True, help="包含summary的hourly_aggregations.jsonl文件")
    parser.add_argument("--collections", nargs="*", help="要分析的collections")
    parser.add_argument("--top-k", type=int, default=5, help="检索top-k个相关entry")
    parser.add_argument("--openai-model", default=DEFAULT_EMBEDDING_MODEL, help="OpenAI embedding模型")
    parser.add_argument("--openai-api-key", default=DEFAULT_API_KEY, help="OpenAI API Key")
    parser.add_argument("--openai-base-url", default=DEFAULT_BASE_URL, help="OpenAI Base URL")
    parser.add_argument("--time-buffer-hours", type=float, default=1.5, help="时间缓冲区：排除summary所在小时前后各N小时的entry（默认1.5小时）")
    parser.add_argument("--search-limit", type=int, default=50, help="检索的初始候选数量（过滤后取top-k，默认50）")
    parser.add_argument("--context-window", type=int, default=3, help="上下文窗口大小，前后各包含的entry数量")
    parser.add_argument("--output", type=Path, required=True, help="保存检索结果的JSON文件路径")
    parser.add_argument("--skip-fact-filter", action="store_true", help="跳过fact类型过滤，返回所有类型的entry（用于entry_type为空的旧数据）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_summary_retrieval(
        args.qdrant_dir,
        args.aggregations,
        collections=args.collections,
        top_k=args.top_k,
        openai_model=args.openai_model,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        time_buffer_hours=args.time_buffer_hours,
        search_limit=args.search_limit,
        context_window=args.context_window,
        skip_fact_filter=args.skip_fact_filter,
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\n共保存 {len(results)} 条 bucket 结果到 {output_path}")


if __name__ == "__main__":
    main()