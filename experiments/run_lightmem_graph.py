#!/usr/bin/env python3
"""
时间表达式提取 + LLM 事件聚合 + 存入Qdrant
精简版 V3: LLM 只生成描述和时间线，Speaker 和 entities 由程序添加
"""

import json
import re
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
import spacy
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI

# 导入您现有的模块
from lightmem.configs.retriever.embeddingretriever.qdrant import QdrantConfig
from lightmem.factory.retriever.embeddingretriever.qdrant import Qdrant

# ============ 配置 ============
DATA_PATH = '/disk/disk_20T/xubuqiang/lightmem/dataset/locomo/locomo10.json'
QDRANT_DIR = './qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_512_0_8_pre_update_adding_timestamp'
OUTPUT_QDRANT_DIR = './qdrant_temporal_events_v3'
OUTPUT_JSON_DIR = './temporal_events_json_v3'

# LLM API 配置
API_KEY = 'sk-elh8U89D8GUNipT9CdDaF39a1e5d44649e48B392E98cC24d'
API_BASE_URL = 'https://api.gpts.vin/v1'
LLM_MODEL = 'gpt-4o-mini'

os.makedirs(OUTPUT_QDRANT_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

# 初始化 OpenAI 客户端
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ============ LLM Prompt模板 ============
EVENT_AGGREGATION_PROMPT = """You are an AI assistant that analyzes conversation entries for a SINGLE speaker and groups them into distinct events/themes.

Below are all conversation entries for this speaker, sorted chronologically:

{entries}

Your task:
1. Identify different event themes or topics this person discussed
2. Group entries into separate events if they belong to clearly different themes
3. For each event group, create an Event_description and Event_timeline

Output format (JSON):
{{
  "data": [
    {{
      "Event_description": "a brief natural description of this event theme",
      "Event_timeline": [
        "YYYY-MM-DD mentioned that [speaker_name] [action] on/at [time]",
        ...
      ]
    }},
    {{
      "Event_description": "a brief natural description of another event theme",
      "Event_timeline": [...]
    }}
  ]
}}

Example - Input:
[2023-05-25 13:14:00, Thu] Alice ran a charity race for mental health last Saturday. [mention 2023-05-25, ran on 2023-05-20]
[2023-06-10 14:20:00, Sat] Alice started volunteering at animal shelter yesterday.
[2023-06-15 10:30:00, Thu] Alice adopted a dog from the shelter last week.
[2023-07-03 13:36:03, Mon] Alice signed up for a pottery class yesterday.
[2023-07-15 13:51:01, Sat] Alice took her kids to a pottery workshop last Friday.
[2023-08-25 13:33:03, Fri] Alice made a plate in pottery class yesterday. [mention 2023-08-25, made plate on 2023-08-24]

Example - Output:
{{
  "data": [
    {{
      "Event_description": "Alice participated in charity race event focused on mental health awareness",
      "Event_timeline": [
        "2023-05-25 mentioned that Alice ran a charity race for mental health on 2023-05-20"
      ]
    }},
    {{
      "Event_description": "Alice started volunteering at animal shelter and adopted a dog",
      "Event_timeline": [
        "2023-06-10 mentioned that Alice started volunteering at animal shelter yesterday",
        "2023-06-15 mentioned that Alice adopted a dog from the shelter last week"
      ]
    }},
    {{
      "Event_description": "Alice took pottery classes and attended pottery workshop with kids",
      "Event_timeline": [
        "2023-07-03 mentioned that Alice signed up for a pottery class yesterday",
        "2023-07-15 mentioned that Alice took her kids to a pottery workshop last Friday",
        "2023-08-25 mentioned that Alice made a plate in pottery class on 2023-08-24"
      ]
    }}
  ]
}}

Important guidelines:
- Identify distinct themes: charity/volunteering, hobbies, family, work, health, etc.
- Create separate event groups for clearly different topics
- If entries are closely related (like pottery class and pottery workshop), group them together
- Event_description: A natural language description (not just keywords) of what this event theme is about
- Event_timeline: Format "mention_date mentioned that [speaker_name] [action] on/at [specific_time]"
- Use time annotations from entries like "[mention YYYY-MM-DD, event on YYYY-MM-DD]"
- Preserve relative times naturally (yesterday, last week) if no specific date given

Now generate the JSON output:"""


# ============ 时间表达式提取器 ============
class TimeExpressionExtractor:
    """时间表达式提取器"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # 复合时间表达式
        self.compound_patterns = [
            (r'\bthe\s+(week|month|year|weekend|day)\s+before\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+[A-Z][a-z]+,?\s+\d{4}|\d{4}-\d{2}-\d{2})', 'compound_before'),
            (r'\b(\d+|a|one|two|three|four|five)\s+(weeks?|months?|days?|weekends?)\s+before\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+[A-Z][a-z]+,?\s+\d{4}|\d{4}-\d{2}-\d{2})', 'compound_n_before'),
            (r'\bthe\s+(week|month|year|weekend|day)\s+after\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+[A-Z][a-z]+,?\s+\d{4}|\d{4}-\d{2}-\d{2})', 'compound_after'),
            (r'\b(first|second|third|fourth|last)\s+(week|weekend|day)\s+of\s+([A-Z][a-z]+),?\s+(\d{4})', 'compound_ordinal_of'),
            (r'\bthe\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+before\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+[A-Z][a-z]+,?\s+\d{4}|\d{4}-\d{2}-\d{2})', 'compound_weekday_before'),
        ]
        
        # 基础时间表达式
        self.basic_patterns = [
            (r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|weekend|weekends)\b', 'weekday'),
            (r'\b(yesterday|today|tomorrow)\b', 'relative_day'),
            (r'\b(last|next|this)\s+(week|month|year|weekend|summer|winter|spring|fall|autumn|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b', 'relative_period'),
            (r'\b(\d+|a|one|two|three|four|five|several)\s+(days?|weeks?|months?|years?|weekends?)\s+(ago|later)\b', 'relative_offset'),
            (r'\bin\s+(\d+|a|one|two|three|four|five)\s+(days?|weeks?|months?|years?|weekends?)\b', 'future_offset'),
            (r'\b\d{4}-\d{2}-\d{2}\b', 'iso_date'),
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', 'full_date'),
            (r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b', 'full_date_alt'),
            (r'\b(19|20)\d{2}\b', 'year_only'),
            (r'\b(first|second|third|last)\s+(time|week|month|year)\b', 'ordinal'),
            (r'\bfor\s+(\d+|a|one|two|several)\s+(years?|months?|weeks?)\b', 'duration_years'),
            (r'\bsince\s+(last|this|yesterday)\s+(week|month|year|weekend)\b', 'since'),
            (r'\b(\d+)\s+(years?|months?|weeks?)\s+ago\b', 'time_ago'),
        ]
        
        self.all_patterns = (
            [(re.compile(p, re.IGNORECASE), t) for p, t in self.compound_patterns] +
            [(re.compile(p, re.IGNORECASE), t) for p, t in self.basic_patterns]
        )
    
    def extract_from_text(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取时间表达式"""
        results = []
        
        # 正则匹配
        for pattern, expr_type in self.all_patterns:
            for match in pattern.finditer(text):
                results.append({
                    'expression': match.group(0),
                    'type': expr_type,
                    'span': match.span(),
                })
        
        # spaCy DATE实体
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                results.append({
                    'expression': ent.text,
                    'type': 'spacy_date',
                    'span': (ent.start_char, ent.end_char),
                })
        
        # 去重
        results = self._deduplicate_expressions(results)
        return results
    
    def _deduplicate_expressions(self, expressions: List[Dict]) -> List[Dict]:
        """去重：优先保留compound类型"""
        if not expressions:
            return []
        
        type_priority = {
            'compound_before': 10, 'compound_n_before': 10, 'compound_after': 10,
            'compound_ordinal_of': 10, 'compound_weekday_before': 10,
            'full_date': 8, 'full_date_alt': 8, 'iso_date': 8,
            'time_ago': 7, 'duration_years': 7,
            'relative_period': 6, 'relative_offset': 6,
            'weekday': 5, 'spacy_date': 3,
        }
        
        expressions.sort(key=lambda x: (x['span'][0], -x['span'][1]))
        
        result = []
        for expr in expressions:
            overlaps = False
            for existing in result:
                if self._spans_overlap(expr['span'], existing['span']):
                    overlaps = True
                    expr_priority = type_priority.get(expr['type'], 0)
                    existing_priority = type_priority.get(existing['type'], 0)
                    
                    if expr_priority > existing_priority:
                        result.remove(existing)
                        result.append(expr)
                    elif expr_priority == existing_priority:
                        expr_len = expr['span'][1] - expr['span'][0]
                        existing_len = existing['span'][1] - existing['span'][0]
                        if expr_len > existing_len:
                            result.remove(existing)
                            result.append(expr)
                    break
            
            if not overlaps:
                result.append(expr)
        
        return result
    
    @staticmethod
    def _spans_overlap(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        return not (span1[1] <= span2[0] or span2[1] <= span1[0])


# ============ LLM 调用函数 ============
def call_llm_for_single_speaker(speaker_name: str, temporal_entries: List[Dict], max_retries: int = 3) -> Dict:
    """调用 LLM 为单个 speaker 进行事件聚合"""
    
    # 格式化输入
    entries_text = format_entries_for_llm(temporal_entries)
    prompt = EVENT_AGGREGATION_PROMPT.format(entries=entries_text)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes conversations and outputs structured JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            
            # 提取响应内容
            content = response.choices[0].message.content.strip()
            
            # 尝试解析 JSON
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            
            # 程序添加 Speaker 字段
            for event in result.get('data', []):
                event['Speaker'] = speaker_name
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"    ⚠ JSON parsing error for {speaker_name} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print(f"    ✗ Failed to parse LLM response for {speaker_name}")
                return {"data": []}
        except Exception as e:
            print(f"    ⚠ LLM API error for {speaker_name} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print(f"    ✗ Failed to call LLM for {speaker_name}")
                return {"data": []}
    
    return {"data": []}


def format_entries_for_llm(temporal_entries: List[Dict]) -> str:
    """格式化entries为LLM输入格式"""
    lines = []
    
    for entry in temporal_entries:
        payload = entry.get('payload', {})
        
        time_stamp = payload.get('time_stamp', '')
        weekday = payload.get('weekday', '')
        speaker = payload.get('speaker_name', 'Unknown')
        memory = payload.get('memory', '')
        
        # 格式化时间戳
        if time_stamp:
            try:
                dt = datetime.fromisoformat(time_stamp)
                formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                formatted_time = time_stamp
        else:
            formatted_time = 'N/A'
        
        lines.append(f"[{formatted_time}, {weekday}] {speaker} {memory}")
    
    return "\n".join(lines)


# ============ 按 Speaker 分组 ============
def group_entries_by_speaker(temporal_entries: List[Dict]) -> Dict[str, List[Dict]]:
    """按 speaker 分组并排序"""
    speaker_groups = {}
    
    for entry in temporal_entries:
        speaker = entry.get('payload', {}).get('speaker_name', 'Unknown')
        if speaker not in speaker_groups:
            speaker_groups[speaker] = []
        speaker_groups[speaker].append(entry)
    
    # 每个 speaker 的 entries 按时间排序
    for speaker in speaker_groups:
        speaker_groups[speaker].sort(
            key=lambda x: datetime.fromisoformat(x.get('payload', {}).get('time_stamp', '1970-01-01T00:00:00'))
        )
    
    return speaker_groups


# ============ Qdrant 操作函数 ============
def load_entries_from_qdrant(sample_id: str, qdrant_dir: str) -> List[Dict]:
    """从Qdrant加载entries"""
    try:
        config = QdrantConfig(
            collection_name=sample_id,
            path=f'{qdrant_dir}/{sample_id}',
            embedding_model_dims=1536,
            on_disk=True,
        )
        retriever = Qdrant(config)
        entries = retriever.get_all(with_vectors=False, with_payload=True)
        return entries
    except Exception as e:
        print(f"✗ Failed to load {sample_id}: {e}")
        return []


def save_to_qdrant(sample_id: str, event_data: Dict, output_dir: str):
    """将事件数据保存到Qdrant集合"""
    collection_name = f"{sample_id}_events"
    collection_path = os.path.join(output_dir, collection_name)
    
    # 创建Qdrant client
    client_qdrant = QdrantClient(path=collection_path)
    
    # 创建集合
    try:
        client_qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=128, distance=Distance.COSINE),
        )
    except Exception as e:
        pass  # Collection might already exist
    
    # 准备points
    points = []
    
    for idx, event in enumerate(event_data.get('data', [])):
        # 将 Event_timeline 转换为字符串
        timeline = event.get('Event_timeline', [])
        if isinstance(timeline, list):
            timeline_str = '\n'.join(timeline)
        else:
            timeline_str = str(timeline)
        
        points.append(PointStruct(
            id=idx,
            vector=[0.0] * 128,  # dummy vector
            payload={
                'sample_id': sample_id,
                'speaker': event.get('Speaker', ''),
                'event_description': event.get('Event_description', ''),
                'event_timeline': timeline_str,
                'created_at': datetime.now().isoformat(),
            }
        ))
    
    if points:
        client_qdrant.upsert(
            collection_name=collection_name,
            points=points,
        )
        print(f"  ✓ Saved to Qdrant: {collection_path} ({len(points)} events)")


def save_to_json(sample_id: str, event_data: Dict, output_dir: str):
    """保存事件数据到JSON文件"""
    output_file = os.path.join(output_dir, f'{sample_id}_events.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(event_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved to JSON: {output_file}")


# ============ 核心处理函数 ============
def process_sample(sample_id: str, qdrant_dir: str, output_qdrant_dir: str, output_json_dir: str, extractor):
    """处理单个sample - 按speaker分别调用LLM"""
    print(f"\n{'='*80}")
    print(f"Processing: {sample_id}")
    print(f"{'='*80}")
    
    # 1. 加载entries
    entries = load_entries_from_qdrant(sample_id, qdrant_dir)
    if not entries:
        return None
    
    print(f"Loaded {len(entries)} entries")
    
    # 2. 提取包含时间表达式的entries
    temporal_entries = []
    for entry in entries:
        payload = entry.get('payload', {})
        memory = payload.get('memory', '')
        
        time_expressions = extractor.extract_from_text(memory)
        
        if time_expressions:
            temporal_entries.append(entry)
    
    print(f"Found {len(temporal_entries)} temporal entries ({len(temporal_entries)/len(entries)*100:.1f}%)")
    
    if not temporal_entries:
        print("  No temporal entries found, skipping...")
        return None
    
    # 3. 按 speaker 分组并排序
    speaker_groups = group_entries_by_speaker(temporal_entries)
    print(f"Found {len(speaker_groups)} speakers: {list(speaker_groups.keys())}")
    
    # 4. 为每个 speaker 分别调用 LLM
    all_events = []
    for speaker_name, speaker_entries in speaker_groups.items():
        print(f"\n  Processing {speaker_name} ({len(speaker_entries)} entries)...")
        
        # 调用 LLM
        event_data = call_llm_for_single_speaker(speaker_name, speaker_entries)
        
        if event_data.get('data'):
            all_events.extend(event_data['data'])
            print(f"    ✓ Generated {len(event_data['data'])} event group(s) for {speaker_name}")
        else:
            print(f"    ✗ No events generated for {speaker_name}")
    
    if not all_events:
        print(f"  ✗ No events generated for any speaker")
        return None
    
    # 5. 合并结果
    combined_result = {"data": all_events}
    print(f"\n  Total events generated: {len(all_events)}")
    
    # 6. 保存结果
    save_to_qdrant(sample_id, combined_result, output_qdrant_dir)
    save_to_json(sample_id, combined_result, output_json_dir)
    
    return {
        'sample_id': sample_id,
        'total_entries': len(entries),
        'temporal_entries': len(temporal_entries),
        'num_speakers': len(speaker_groups),
        'event_groups': len(all_events),
    }


# ============ 主函数 ============
def main():
    print("="*80)
    print("Temporal Event Extraction & Aggregation (v3 - Simplified)")
    print("="*80)
    
    # 初始化
    print("\nInitializing...")
    extractor = TimeExpressionExtractor()
    
    # 加载数据集
    print(f"\nLoading dataset from {DATA_PATH}")
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    
    # 处理所有samples
    results = []
    for sample in tqdm(data, desc="Processing samples"):
        sample_id = sample['sample_id']
        result = process_sample(
            sample_id, 
            QDRANT_DIR, 
            OUTPUT_QDRANT_DIR,
            OUTPUT_JSON_DIR,
            extractor
        )
        if result:
            results.append(result)
    
    # 打印总结
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Processed {len(results)} samples")
    print(f"Total temporal entries: {sum(r['temporal_entries'] for r in results)}")
    print(f"Total event groups: {sum(r['event_groups'] for r in results)}")
    print(f"Average events per sample: {sum(r['event_groups'] for r in results) / len(results):.1f}")
    print(f"\n✓ Results saved to:")
    print(f"  - Qdrant: {OUTPUT_QDRANT_DIR}/")
    print(f"  - JSON: {OUTPUT_JSON_DIR}/")
    print("="*80)
    
    # 保存统计信息
    stats_file = os.path.join(OUTPUT_JSON_DIR, '_processing_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_samples': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"\n✓ Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()