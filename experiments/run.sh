# nohup python run_lightmem_qwen.py > 2025_10_26_test_no_summary_number_75_data.log 2>&1 &
# nohup python run_bipartite_retrieval.py > 2025_10_26_test_bipartite_retrieval.log 2>&1 &
# nohup python run_hybrid_retrieval.py > 2025_10_26_test_hybrid_retrieval.log 2>&1 &
# nohup python run_vector_retrieval.py > 2025_10_26_test_vector_retrieval.log 2>&1 &
# nohup python run_lightmem_qwen_locomo.py > 2025_10_29_build_locomo_dataset_text-embedding-3-small.log 2>&1 &
# nohup python run_vector_retrieval_locomo_one.py \
#     --dataset /disk/disk_20T/xubuqiang/lightmem/dataset/locomo/locomo10.json \
#     --qdrant-dir ./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_512_0_7 \
#     --output-dir ./results/qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_512_0_7 \
#     --embedder openai \
#     > 2025_11_1_vector_retrieval_locomo_gpt_openai_embedding_one_60.log 2>&1 &

# nohup python run_vector_retrieval_locomo.py \
#     --dataset /disk/disk_20T/xubuqiang/lightmem/dataset/locomo/locomo10.json \
#     --qdrant-dir ./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_256_0_6_post_update \
#     --output-dir ./results/qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_256_0_6_post_update \
#     --embedder openai \
#     > 2025_11_2_vector_retrieval_locomo_gpt_openai_embedding_two_60_post_update_256_0_6.log 2>&1 &


# nohup python run_vector_retrieval_locomo.py \
#     --dataset /disk/disk_20T/xubuqiang/lightmem/dataset/locomo/locomo10.json \
#     --qdrant-dir ./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_256_0_7_post_update \
#     --output-dir ./results/qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_256_0_7_post_update \
#     --embedder openai \
#     > 2025_11_2_vector_retrieval_locomo_gpt_openai_embedding_two_60_post_update_256_0_7.log 2>&1 &


# nohup python run_vector_retrieval_locomo.py \
#     --dataset /disk/disk_20T/xubuqiang/lightmem/dataset/locomo/locomo10.json \
#     --qdrant-dir ./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_256_0_8_post_update \
#     --output-dir ./results/qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_256_0_8_post_update \
#     --embedder openai \
#     > 2025_11_2_vector_retrieval_locomo_gpt_openai_embedding_two_60_post_update_256_0_8.log 2>&1 &

# nohup python run_vector_retrieval_locomo.py \
#     --dataset /disk/disk_20T/xubuqiang/lightmem/dataset/locomo/locomo10.json \
#     --qdrant-dir ./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_256_0_6_pre_update \
#     --output-dir ./results/qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_256_0_6_pre_update \
#     --embedder openai \
#     > 2025_11_2_vector_retrieval_locomo_gpt_openai_embedding_two_60_pre_update_256_0_6.log 2>&1 &


# nohup python run_vector_retrieval_locomo.py \
#     --dataset /disk/disk_20T/xubuqiang/lightmem/dataset/locomo/locomo10.json \
#     --qdrant-dir ./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_256_0_7_pre_update \
#     --output-dir ./results/qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_256_0_7_pre_update \
#     --embedder openai \
#     > 2025_11_2_vector_retrieval_locomo_gpt_openai_embedding_two_60_pre_update_256_0_7.log 2>&1 &


# nohup python run_vector_retrieval_locomo.py \
#     --dataset /disk/disk_20T/xubuqiang/lightmem/dataset/locomo/locomo10.json \
#     --qdrant-dir ./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_256_0_8_pre_update \
#     --output-dir ./results/qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_256_0_8_pre_update \
#     --embedder openai \
#     > 2025_11_2_vector_retrieval_locomo_gpt_openai_embedding_two_60_pre_update_256_0_8.log 2>&1 &


# nohup python run_vector_retrieval_locomo.py \
#     --dataset /disk/disk_20T/xubuqiang/lightmem/dataset/locomo/locomo10.json \
#     --qdrant-dir ./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_768_0_6_post_update \
#     --output-dir ./results/qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_768_0_6_post_update \
#     --embedder openai \
#     > 2025_11_2_vector_retrieval_locomo_gpt_openai_embedding_two_60_post_update_768_0_6.log 2>&1 &


# nohup python run_vector_retrieval_locomo.py \
#     --dataset /disk/disk_20T/xubuqiang/lightmem/dataset/locomo/locomo10.json \
#     --qdrant-dir ./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_768_0_7_post_update \
#     --output-dir ./results/qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_768_0_7_post_update \
#     --embedder openai \
#     > 2025_11_2_vector_retrieval_locomo_gpt_openai_embedding_two_60_post_update_768_0_7.log 2>&1 &


# nohup python run_vector_retrieval_locomo.py \
#     --dataset /disk/disk_20T/xubuqiang/lightmem/dataset/locomo/locomo10.json \
#     --qdrant-dir ./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_768_0_8_post_update \
#     --output-dir ./results/qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_768_0_8_post_update \
#     --embedder openai \
#     > 2025_11_2_vector_retrieval_locomo_gpt_openai_embedding_two_60_post_update_768_0_8.log 2>&1 &

# nohup python run_vector_retrieval_locomo.py \
#     --dataset /disk/disk_20T/xubuqiang/lightmem/dataset/locomo/locomo10.json \
#     --qdrant-dir ./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_768_0_6_pre_update \
#     --output-dir ./results/qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_768_0_6_pre_update \
#     --embedder openai \
#     > 2025_11_2_vector_retrieval_locomo_gpt_openai_embedding_two_60_pre_update_768_0_6.log 2>&1 &


# nohup python run_vector_retrieval_locomo.py \
#     --dataset /disk/disk_20T/xubuqiang/lightmem/dataset/locomo/locomo10.json \
#     --qdrant-dir ./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_768_0_7_pre_update \
#     --output-dir ./results/qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_768_0_7_pre_update \
#     --embedder openai \
#     > 2025_11_2_vector_retrieval_locomo_gpt_openai_embedding_two_60_pre_update_768_0_7.log 2>&1 &


# nohup python run_vector_retrieval_locomo.py \
#     --dataset /disk/disk_20T/xubuqiang/lightmem/dataset/locomo/locomo10.json \
#     --qdrant-dir ./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_768_0_8_pre_update \
#     --output-dir ./results/qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_768_0_8_pre_update \
#     --embedder openai \
#     > 2025_11_2_vector_retrieval_locomo_gpt_openai_embedding_two_60_pre_update_768_0_8.log 2>&1 &


# python download_offline_resources.py \
#   --models roberta-large \
#   --nltk-resources punkt_tab

# nohup python run_unified_retrieval.py --mode vector > 2025_10_29_longmemeval_unified_vector.log 2>&1 &
# nohup python run_unified_retrieval.py --mode bipartite > 2025_10_29_longmemeval_unified_bipartite.log 2>&1 &
# nohup python run_unified_retrieval.py --mode hybrid > 2025_10_29_longmemeval_unified_hybrid.log 2>&1 &

# update qdrant vectors using openai embeddings
# nohup python update_qdrant_vectors.py --qdrant-root ./qdrant_data_locomo --apply > 2025_10_29_update_qdrant_vectors.log 2>&1 &

# nohup python run_lightmem_qwen_locomo.py  > 2025_10_30_build_locomo_dataset_gpt-4o-mini_all-MiniLM-L6-v2_512_0_7.log 2>&1 &
# nohup python run_lightmem_qwen_locomo.py  > 2025_10_31_build_locomo_dataset_gpt-4o-mini_text-embedding-3-small_512_0_7.log 2>&1 &
# nohup python run_lightmem_qwen_locomo.py  > 2025_11_1_build_locomo_dataset_gpt-4o-mini_text-embedding-3-small_512_0_7_split_add_and_update.log 2>&1 &
# nohup python run_lightmem_qwen_locomo.py  > 2025_11_1_build_locomo_dataset_gpt-4o-mini_text-embedding-3-small_512_0_6_split_add_and_update.log 2>&1 &
# nohup python run_lightmem_qwen_locomo.py  > 2025_11_2_build_locomo_dataset_gpt-4o-mini_text-embedding-3-small_768_0_7_split_add_and_update_2.log 2>&1 &
# nohup python run_lightmem_qwen_locomo.py  > 2025_11_2_build_locomo_dataset_gpt-4o-mini_text-embedding-3-small_768_0_6_split_add_and_update_2.log 2>&1 &
# nohup python run_lightmem_qwen_locomo.py  > 2025_11_2_build_locomo_dataset_gpt-4o-mini_text-embedding-3-small_768_0_8_split_add_and_update_2.log 2>&1 &
# nohup python run_lightmem_qwen_locomo.py  > 2025_11_2_build_locomo_dataset_gpt-4o-mini_text-embedding-3-small_256_0_8_split_add_and_update_2.log 2>&1 &
# nohup python run_lightmem_qwen_locomo.py  > 2025_11_2_build_locomo_dataset_gpt-4o-mini_text-embedding-3-small_256_0_7_split_add_and_update_2.log 2>&1 &
# nohup python run_lightmem_qwen_locomo.py  > 2025_11_2_build_locomo_dataset_gpt-4o-mini_text-embedding-3-small_256_0_6_split_add_and_update_2.log 2>&1 &
# nohup python run_lightmem_qwen_locomo.py  > 2025_11_2_build_locomo_dataset_gpt-4o-mini_text-embedding-3-small_512_0_7_split_add_and_update.log 2>&1 &
# nohup python run_lightmem_qwen_locomo.py  > 2025_11_2_build_locomo_dataset_gpt-4o-mini_text-embedding-3-small_512_0_8_split_add_and_update.log 2>&1 &

# nohup python run_lightmem_qwen_locomo_parallel.py  > 2025_11_3_build_locomo_dataset_gpt-4o-mini_text-embedding-3-small_512_0_8_parallel_prompt_v4.log 2>&1 &


# nohup python run_lightmem_qwen_locomo_parallel.py  > 2025_11_8_build_locomo_dataset_gpt-4o-mini_text-embedding-3-small_512_0_8_split_add_and_update_prompt_locomo_test.log 2>&1 &

# nohup python build_graph.py \
#   --qdrant-dir /disk/disk_20T/xubuqiang/lightmem/experiments/qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_512_0_8_pre_update_prompt_v7 \
#   --dbscan-eps 0.25 \
#   --dbscan-min-samples 3 \
#   --context-window 5 \
#   --min-cluster-size 4 \
#   > 2025_11_9_build_graph_test.log 2>&1 &

# nohup python inspect_clusters.py \
#   --qdrant-dir /disk/disk_20T/xubuqiang/lightmem/experiments/qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_512_0_8_pre_update_prompt_v7 \
#   --collections conv-26 \
#   --dbscan-eps 0.25 \
#   --dbscan-min-samples 2 \
#   > 2025_11_9_clusters.log 2>&1 &

# nohup python aggregate_by_hour.py \
#   --qdrant-dir /disk/disk_20T/xubuqiang/lightmem/experiments/qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_512_0_8_pre_update_prompt_v7 \
#   --collections conv-26 \
#   --output hourly_conv26.jsonl \
#   --use-openai-embedder \
#   --openai-model text-embedding-3-small \
#   --openai-api-key sk-elh8U89D8GUNipT9CdDaF39a1e5d44649e48B392E98cC24d \
#   --openai-base-url https://api.gpts.vin/v1 \
#   > 2025_11_9_aggregate_by_hour.log 2>&1 &

# nohup python cluster_hourly_buckets.py \
# 	--input hourly_conv26.jsonl \
# 	--dbscan-eps 0.22 \
# 	--dbscan-min-samples 2 \
# 	--print-members \
# 	--output clustered_hourly_conv26.jsonl \
# 	> 2025_11_9_cluster_hourly.log 2>&1 &


# nohup python aggregate_one_hour.py \
#   --qdrant-dir /disk/disk_20T/xubuqiang/lightmem/experiments/qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_512_0_8_post_update_prompt_with_interaction \
#   --openai-model text-embedding-3-small \
#   --openai-api-key "sk-elh8U89D8GUNipT9CdDaF39a1e5d44649e48B392E98cC24d" \
#   --openai-base-url https://api.gpts.vin/v1 \
#   --output hourly_clustered_with_interaction.jsonl \
#   > 2025_11_10_cluster_hourly.log 2>&1 &


# nohup python run_vector_retrieval_locomo.py \
#     --dataset /disk/disk_20T/xubuqiang/lightmem/dataset/locomo/locomo10.json \
#     --qdrant-dir ./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_512_0_8_pre_update_prompt_with_interaction_v6 \
#     --aggregations-path ./hourly_with_summaries_v6.jsonl \
#     --output-dir ./results/qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_512_0_8_pre_update_combined_60_with_interaction_and_summary_v6_0_3_11_07_test \
#     --total-limit 60 \
#     --retrieval-mode combined \
#     --embedder openai \
#     --similarity-threshold 0.3 \
#     > 2025_11_12_vector_retrieval_locomo_gpt_openai_embedding_combined_60_pre_update_adding_timestamp_512_0_8_pre_update_locomo_test_with_interaction_and_summary_v6_0_3_11_07_test.log 2>&1 &

# nohup python run_lightmem_qwen_locomo_parallel.py  > 2025_11_11_build_locomo_dataset_gpt-4o-mini_text-embedding-3-small_512_0_8_with_interaction_v6.log 2>&1 &

nohup python run_summary.py \
  --qdrant-dir ./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_512_0_8_pre_update_prompt_with_interaction_v6 \
  --output hourly_with_summaries_count_runtime.jsonl \
  > 2025_11_12_hourly_with_summaries_count_runtime.log 2>&1 &

# nohup python summary_retrieval.py \
#   --qdrant-dir ./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_512_0_8_pre_update_prompt_with_interaction_v6 \
#   --aggregations hourly_with_summaries_v6.jsonl \
#   --output summaries_retrieval_v6.jsonl \
#   > 2025_11_11_analyze_summary_retrieval_v6.log 2>&1 &

# nohup python build_graph.py \
#   --qdrant-dir ./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_512_0_8_pre_update_prompt_with_interaction_v6 \
#   --aggregations ./hourly_with_summaries_v6.jsonl \
#   --retrieval-output ./retrieval_results_v6_6.json \
#   --response-output ./llm_responses_v6_6.json \
#   --relations-output ./graph_v6_6.json \
#   --events-output ./events_data_v6_6.json \
#   > 2025_11_12_build_graph_v6_6.log 2>&1 &

# nohup python inspect_graph_structure.py \
#   graph_v6_4.json \
#   --limit 1 \
#   > 2025_11_12_inspect_graph_structure_v6_4.log 2>&1 & 
 
# nohup python summary_retrieval_and_build_graph.py \
#   --input summaries_retrieval_v4.jsonl \
#   > 2025_11_11_analyze_summary_retrieval_and_build_graph_v4.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python run_lightmem_qwen_locomo_parallel.py  > 2025_11_12_build_locomo_dataset_gpt-4o-mini_text-embedding-3-small_512_0_8_collect_token_cunsumption.log 2>&1 &
