# nohup python add.py > locomo_add.log 2>&1 &


# nohup python search.py \
#     --dataset /disk/disk_20T/xubuqiang/lightmem/dataset/locomo/locomo10.json \
#     --qdrant-dir ./qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_512_0_8_pre_update_prompt_v3 \
#     --output-dir ./results/qdrant_data_locomo_gpt-4o-mini_text-embedding-3-small_512_0_8_pre_update_prompt_v3_two_60_change_memory_timestamp \
#     --limit-per-speaker 60 \
#     --embedder openai \
#     > locomo_search.log 2>&1 &
