# 把entries从Qdrant导出到本地文件夹，提取terms，排除动词
# python export_entries.py \
#     --collection e47becba \
#     --qdrant-path /disk/disk_20T/xubuqiang/lightmem/pre_experiments/test/qdrant_data \
#     --out-dir /disk/disk_20T/xubuqiang/lightmem/pre_experiments/test/exports \
#     --extract-terms --spacy-model en_core_web_sm --exclude-verbs 
# 提取query中的terms，排除动词
# python query_entities.py \
#     --input-file /disk/disk_20T/xubuqiang/lightmem/pre_experiments/test/longmemeval_questions.json \
#     --output-dir /disk/disk_20T/xubuqiang/lightmem/pre_experiments/test \
#     --model en_core_web_sm --exclude-verbs
# 构建bipartite图
# python build_term_bipartite.py \
#   --entries ./exports/e47becba_entries.json \
#   --out-dir ./output \
#   --query-terms "Netflix" \
#   --max-hops 3 \
#   --draw-multihop 
# topic_summary和query的相似度rank排序
python rank_topic_summary.py \
    --json /disk/disk_20T/xubuqiang/lightmem/pre_experiments/test/exports/e47becba_entries.json \
    --query "What degree did I graduate with?" \
    --target-topic 58 \
    --top-k 10