import json
from pathlib import Path

input_file = Path("hourly_clustered_with_interaction.jsonl")

if not input_file.exists():
    print(f"文件不存在: {input_file}")
    exit(1)

with open(input_file, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

print(f"总共有 {len(records)} 条记录\n")
print("=" * 100)

for i, record in enumerate(records):
    print(f"\n记录 {i+1}/{len(records)}")
    print("=" * 100)
    print(f"Collection: {record['collection']}")
    print(f"时间桶: {record['bucket']}")
    print(f"开始时间: {record['start_time']}")
    print(f"结束时间: {record['end_time']}")
    print(f"Entry数量: {len(record['entry_ids'])}")
    print(f"参与者: {', '.join(record['speakers'])}")
    print(f"\n聚合文本:")
    print("-" * 100)
    print(record['aggregated_text'])
    print("=" * 100)