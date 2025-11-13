import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists():
    sys.path.append(str(SRC_PATH))

from retrievers import QdrantEntryLoader  # type: ignore[import]


def get_entry_type(entry: Dict[str, Any]) -> str:
    """获取entry的类型"""
    payload = entry.get("payload", {})
    
    # 优先查找 entry_type
    entry_type = payload.get("entry_type") or entry.get("entry_type")
    if entry_type:
        return str(entry_type).lower()
    
    # 其次查找 memory_type
    memory_type = payload.get("memory_type") or entry.get("memory_type")
    if memory_type:
        return str(memory_type).lower()
    
    return "unknown"


def discover_collections(qdrant_dir: Path) -> List[str]:
    """通过扫描目录结构来发现collections"""
    collections = []
    
    # 尝试多种可能的目录结构
    # 结构1: qdrant_dir/collection_name/
    if qdrant_dir.exists():
        for item in qdrant_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                collections.append(item.name)
    
    return sorted(collections)


def analyze_entry_types(qdrant_dir: Path, collections: List[str] = None, debug: bool = False) -> None:
    """分析entry类型分布"""
    if not qdrant_dir.exists():
        raise FileNotFoundError(f"Qdrant目录不存在: {qdrant_dir}")
    
    print(f"Qdrant目录: {qdrant_dir}\n")
    
    if debug:
        print("调试模式: 显示目录详细信息")
        print(f"目录绝对路径: {qdrant_dir.absolute()}")
        print(f"目录内容:")
        for item in sorted(qdrant_dir.iterdir()):
            item_type = "目录" if item.is_dir() else "文件"
            print(f"  - {item.name} ({item_type})")
        print()
    
    loader = QdrantEntryLoader(str(qdrant_dir))
    
    # 获取所有collections - 尝试多种方法
    all_collections = []
    
    # 方法1: 使用loader的list_collections方法
    try:
        all_collections = loader.list_collections()
        print(f"通过loader.list_collections()发现 {len(all_collections)} 个collections")
        if debug and all_collections:
            print(f"详细列表: {all_collections}")
    except Exception as e:
        print(f"loader.list_collections()失败: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        # 方法2: 通过目录扫描
        all_collections = discover_collections(qdrant_dir)
        print(f"通过目录扫描发现 {len(all_collections)} 个collections")
    
    if all_collections:
        print(f"Collections: {', '.join(all_collections)}\n")
    else:
        print("未发现任何collections\n")
        print("目录结构:")
        for item in qdrant_dir.iterdir():
            print(f"  - {item.name} ({'目录' if item.is_dir() else '文件'})")
        return
    
    # 过滤collections
    if collections:
        target_collections = [c for c in collections if c in all_collections]
        missing = [c for c in collections if c not in all_collections]
        if missing:
            print(f"警告: 以下collections不存在: {', '.join(missing)}")
        print(f"将分析以下collections: {', '.join(target_collections)}\n")
    else:
        target_collections = all_collections
    
    if not target_collections:
        print("没有可分析的collection")
        return
    
    # 统计每个collection的类型分布
    overall_counter = Counter()
    collection_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
    
    for collection in target_collections:
        print(f"{'='*80}")
        print(f"Collection: {collection}")
        print(f"{'='*80}")
        
        try:
            if debug:
                print(f"尝试加载collection: {collection}")
            entries = loader.load_entries(collection, with_vectors=False)
            if debug:
                print(f"成功加载 {len(entries)} 个entries")
        except Exception as exc:
            print(f"加载失败: {exc}")
            if debug:
                import traceback
                traceback.print_exc()
            print()
            continue
        
        if not entries:
            print("未找到任何entries\n")
            continue
        
        # 统计类型
        type_counter = Counter()
        for entry in entries:
            entry_type = get_entry_type(entry)
            type_counter[entry_type] += 1
            overall_counter[entry_type] += 1
        
        if debug and entries:
            print(f"\n示例entry结构:")
            sample_entry = entries[0]
            print(f"  Keys: {list(sample_entry.keys())}")
            if 'payload' in sample_entry:
                print(f"  Payload keys: {list(sample_entry['payload'].keys())}")
            print()
        
        # 输出该collection的统计
        total = sum(type_counter.values())
        print(f"总entry数: {total}")
        print(f"\n类型分布:")
        
        # 按数量排序
        for entry_type, count in type_counter.most_common():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {entry_type:20s}: {count:6d} ({percentage:5.1f}%)")
        
        collection_stats[collection] = {
            "total": total,
            "types": dict(type_counter),
        }
        print()
    
    # 输出总体统计
    if len(target_collections) > 1:
        print(f"{'='*80}")
        print(f"总体统计 (所有collections)")
        print(f"{'='*80}")
        
        total_entries = sum(overall_counter.values())
        print(f"总entry数: {total_entries}")
        print(f"\n总体类型分布:")
        
        for entry_type, count in overall_counter.most_common():
            percentage = (count / total_entries * 100) if total_entries > 0 else 0
            print(f"  {entry_type:20s}: {count:6d} ({percentage:5.1f}%)")
        print()
    
    # 输出汇总表格
    if len(target_collections) > 1:
        print(f"{'='*80}")
        print(f"Collection汇总表")
        print(f"{'='*80}")
        
        # 获取所有出现过的类型
        all_types = sorted(overall_counter.keys())
        
        # 表头
        header = f"{'Collection':<30s}"
        for t in all_types:
            header += f" | {t:>10s}"
        header += f" | {'Total':>10s}"
        print(header)
        print("-" * len(header))
        
        # 每个collection的数据
        for collection in target_collections:
            stats = collection_stats.get(collection, {})
            types = stats.get("types", {})
            total = stats.get("total", 0)
            
            row = f"{collection:<30s}"
            for t in all_types:
                count = types.get(t, 0)
                row += f" | {count:>10d}"
            row += f" | {total:>10d}"
            print(row)
        
        # 总计行
        total_row = f"{'TOTAL':<30s}"
        for t in all_types:
            count = overall_counter[t]
            total_row += f" | {count:>10d}"
        total_row += f" | {total_entries:>10d}"
        print("-" * len(header))
        print(total_row)


def main():
    parser = argparse.ArgumentParser(
        description="查看Qdrant数据中entry类型的分布统计"
    )
    parser.add_argument(
        "--qdrant-dir",
        type=Path,
        required=True,
        help="Qdrant数据目录路径",
    )
    parser.add_argument(
        "--collections",
        nargs="*",
        help="指定要分析的collections（不指定则分析所有）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="显示详细的调试信息",
    )
    
    args = parser.parse_args()
    
    try:
        analyze_entry_types(args.qdrant_dir, args.collections, args.debug)
    except Exception as exc:
        print(f"错误: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()