import os
import json
import logging
import tempfile
from typing import Optional

import gradio as gr

from lightmem.memory.lightmem import LightMemory
from lightmem.logging_config import init_logging

# ============ Logging Setup (use centralized init) ============
LOG_DIR = os.path.abspath("./logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "lightmem_run.log")

# Allow user to suppress noisy third-party loggers and keep debug logs in file
init_logging(level=logging.DEBUG,
             fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
             filename=LOG_FILE,
             console_level=logging.INFO,
             file_level=logging.DEBUG,
             force=True,
             logger_levels={"lightmem": logging.DEBUG},
             suppress_loggers=["httpcore", "openai", "urllib3"]) 

# Default config values (kept similar to existing example)
DEFAULTS = {
    "api_key": "sk-mYmdqXKCUL9FqNfI27855c29E94d419c995bA6D54c20Af21",
    "api_base_url": "https://api.gpts.vin/v1",
    "llm_model": "qwen3-30b-a3b-instruct-2507",
    "judge_model": "gpt-4o-mini",
    "llmlingua_model_path": "/disk/disk_20T/fangjizhan/models/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    "embedding_model_path": "/disk/disk_20T/fangjizhan/models/all-MiniLM-L6-v2",
    "data_path": "/disk/disk_20T/xubuqiang/lightmem/dataset/longmemeval/longmemeval_s_cleaned.json",
    "update_mode": "offline",
    "log_file": "./logs/lightmem_run.log",
}

# Example session used to show users the expected JSON format for adding memories
DEFAULT_EXAMPLE_OBJ = {
    "timestamp": "2025-01-10",
    "turns": [
        [
            {"role": "user", "content": "My favorite ice cream flavor is pistachio, and my dog's name is Rex."},
            {"role": "assistant", "content": "Got it. Pistachio is a great choice."}
        ]
    ]
}


css_rules = """
/* 1. 核心居中和宽度控制 */
/* 强制 Gradio 根容器居中 */
.gradio-container {
    max-width: 1400px !important; /* 控制最大宽度 */
    width: 95% !important;        /* 确保自适应宽度 */
    margin-left: auto !important; /* 强制左边距自动 */
    margin-right: auto !important; /* 强制右边距自动，实现居中 */
    padding: 20px; 
}

/* 2. 增强布局和确保侧边栏不留空隙 */
.gradio-container .gr-row { 
    display: flex; 
    gap: 1rem; 
    align-items: flex-start;
}
.gradio-container .gr-column { flex: 1 1 0%; }
.gradio-container .gr-column[style*="display: none"] { display: none !important; }

/* 3. 增强 Markdown 标题样式 */
.gradio-container h2 { border-bottom: 2px solid var(--color-border-primary); padding-bottom: 5px; margin-top: 10px; }
"""

def build_config(api_key: str, api_base_url: str, llm_model: str, llmlingua_model_path: str, embedding_model_path: str, update_mode: str):
    config_dict = {
        "pre_compress": True,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": llmlingua_model_path,
                    "device_map": "cuda",
                    "use_llmlingua2": True,
                },
            }
        },
        "topic_segment": True,
        "precomp_topic_shared": True,
        "topic_segmenter": {"model_name": "llmlingua-2"},
        "messages_use": "user_only",
        "metadata_generate": True,
        "text_summary": True,
        "memory_manager": {
            "model_name": "openai",
            "configs": {
                "model": llm_model,
                "api_key": api_key,
                "max_tokens": 16000,
                "openai_base_url": api_base_url,
            }
        },
        "extract_threshold": 0.1,
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": embedding_model_path,
                "embedding_dims": 384,
                "model_kwargs": {"device": "cuda"},
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": "my_long_term_chat",
                "embedding_model_dims": 384,
                "path": "./my_long_term_chat",
            }
        },
        "update": update_mode,
    }
    return config_dict


def create_lightmem_instance(config):
    # Instantiate LightMemory from config
    lm = LightMemory.from_config(config)
    return lm


def start_app():
    css_rules = """
    .gradio-container { max-width: 1400px; width: 100%; }
    .gradio-container .gr-row { display: flex; gap: 1rem; align-items: flex-start; }
    .gradio-container .gr-column { flex: 1 1 0%; }
    /* Ensure invisible columns do not reserve space */
    .gradio-container .gr-column[style*="display: none"] { display: none !important; }
    """

    with gr.Blocks(title="LightMem UI", css=css_rules, theme=gr.themes.Soft()) as demo:
        gr.Markdown("## LightMem Interactive UI\nA simplified interface: Settings (collapsed) → Add Memory & Retrieve (main).")

        # Header with a small settings opener
        with gr.Row():
            gr.Markdown("### Add memories and query retrieval")
            settings_button = gr.Button("Settings")
            toggle_sidebar_btn = gr.Button("Toggle Sidebar")

        # Settings accordion (collapsed by default)
        with gr.Accordion("Settings", open=False) as settings_acc:
            with gr.Row():
                with gr.Column(scale=1):
                    api_key = gr.Textbox(label="API Key (leave empty to use env)", value=DEFAULTS["api_key"], type="password")
                    api_base = gr.Textbox(label="API Base URL", value=DEFAULTS["api_base_url"]) 
                    llm_model = gr.Textbox(label="LLM Model", value=DEFAULTS["llm_model"]) 
                    update_mode = gr.Dropdown(label="Update Mode", choices=["offline", "online"], value=DEFAULTS["update_mode"]) 
                with gr.Column(scale=1):
                    llmlingua_path = gr.Textbox(label="LLMLingua Model Path", value=DEFAULTS["llmlingua_model_path"]) 
                    embedding_path = gr.Textbox(label="Embedding Model Path", value=DEFAULTS["embedding_model_path"]) 
                    log_file_path = gr.Textbox(label="Log file path", value=DEFAULTS["log_file"]) 
                    init_button = gr.Button("Initialize LightMemory")
                    init_status = gr.Textbox(label="Init Status", interactive=False)

        # Main content: two columns — left for add/retrieve, right for controls & logs
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("### Add Memory (paste JSON session or plain text, or upload .json)")
                memory_text = gr.Textbox(label="Memory Text (single message or JSON session)", lines=6)
                memory_file = gr.File(file_types=[".json"], label="Upload JSON file")
                load_example = gr.Button("Load Example JSON")
                example_display = gr.Textbox(label="Example JSON (click Load Example to copy)", value=json.dumps(DEFAULT_EXAMPLE_OBJ, ensure_ascii=False, indent=2), lines=8, interactive=False)
                add_button = gr.Button("Add Memory")
                add_status = gr.Textbox(label="Add Status", interactive=False)

                gr.Markdown("---")
                gr.Markdown("### Retrieve")
                query = gr.Textbox(label="Query", lines=2)
                k = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Limit")
                retrieve_button = gr.Button("Retrieve")
                retrieve_out = gr.Textbox(label="Retrieve Output", interactive=False, lines=8)

            with gr.Column(scale=2, visible=False) as controls_col:
                gr.Markdown("### Controls & Logs")
                with gr.Row():
                    construct_button = gr.Button("Construct Update Queue")
                    offline_button = gr.Button("Run Offline Update")
                offline_status = gr.Textbox(label="Offline Update Status", interactive=False)
                gr.Markdown("#### Recent Logs")
                log_text = gr.Textbox(label="Logs (tail)", lines=20, interactive=False)
                refresh_logs = gr.Button("Refresh Logs")

        # Internal state
        lm_state = gr.State(value=None)
        settings_open = gr.State(value=False)
        sidebar_open = gr.State(value=False)

        def on_init(api_key_v, api_base_v, llm_model_v, llmlingua_v, embedding_v, update_v):
            cfg = build_config(api_key_v or os.environ.get("OPENAI_API_KEY", ""), api_base_v, llm_model_v, llmlingua_v, embedding_v, update_v)
            try:
                lm = create_lightmem_instance(cfg)
                return lm, "Initialized successfully"
            except Exception as e:
                return None, f"Init failed: {e}"

        init_button.click(on_init, inputs=[api_key, api_base, llm_model, llmlingua_path, embedding_path, update_mode], outputs=[lm_state, init_status])

        # Toggle settings accordion when settings_button clicked
        def toggle_settings(is_open):
            return gr.update(open=not is_open), (not is_open)

        settings_button.click(toggle_settings, inputs=[settings_open], outputs=[settings_acc, settings_open])

        # Toggle sidebar visibility
        def toggle_sidebar_fn(is_open):
            return gr.update(visible=not is_open), (not is_open)

        toggle_sidebar_btn.click(toggle_sidebar_fn, inputs=[sidebar_open], outputs=[controls_col, sidebar_open])

        def on_add_memory(lm, text, uploaded_file):
            if lm is None:
                return "LightMemory not initialized"

            try:
                # If file uploaded, load JSON; expect array of sessions or single session
                if uploaded_file:
                    tmp_path = uploaded_file.name
                    with open(tmp_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                else:
                    # try to parse as JSON first
                    try:
                        payload = json.loads(text)
                    except Exception:
                        # treat as plain user message
                        payload = {"timestamp": "manual", "turns": [[[{"role": "user", "content": text}]]]}

                # Normalize payload: accept dict(session) or list(sessions)
                sessions = []
                if isinstance(payload, list):
                    sessions = payload
                elif isinstance(payload, dict) and "turns" in payload:
                    sessions = [payload]
                else:
                    return "Unsupported JSON format: expected session dict or list of sessions"

                # Add memories
                for session in sessions:
                    ts = session.get("timestamp", "manual")
                    for turn_messages in session.get("turns", []):
                        for msg in turn_messages:
                            if "time_stamp" not in msg:
                                msg["time_stamp"] = ts
                        lm.add_memory(messages=turn_messages, force_segment=True, force_extract=True)

                return f"Added {len(sessions)} session(s)"
            except Exception as e:
                return f"Add failed: {e}"

        add_button.click(on_add_memory, inputs=[lm_state, memory_text, memory_file], outputs=[add_status])

        def on_load_example():
            # Return pretty JSON string to populate the memory_text box
            return json.dumps(DEFAULT_EXAMPLE_OBJ, ensure_ascii=False, indent=2)

        load_example.click(on_load_example, inputs=[], outputs=[memory_text])

        def read_logs(path, n_lines=200):
            try:
                if not path:
                    return "No log path provided"
                if not os.path.exists(path):
                    return f"Log file not found: {path}"
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.read().splitlines()
                tail = lines[-n_lines:]
                return "\n".join(tail)
            except Exception as e:
                return f"Failed to read logs: {e}"

        refresh_logs.click(lambda p: read_logs(p), inputs=[log_file_path], outputs=[log_text])

        def on_construct(lm):
            if lm is None:
                return "LightMemory not initialized"
            try:
                lm.construct_update_queue_all_entries()
                return "Constructed update queue for all entries"
            except Exception as e:
                return f"Construct failed: {e}"

        construct_button.click(on_construct, inputs=[lm_state], outputs=[offline_status])

        def on_offline_update(lm):
            if lm is None:
                return "LightMemory not initialized"
            try:
                lm.offline_update_all_entries(score_threshold=0.8)
                return "Offline update completed"
            except Exception as e:
                return f"Offline update failed: {e}"

        offline_button.click(on_offline_update, inputs=[lm_state], outputs=[offline_status])

        def on_retrieve(lm, q, limit):
            if lm is None:
                return "LightMemory not initialized"
            try:
                results = lm.retrieve(q, limit=limit)
                return json.dumps(results, ensure_ascii=False, indent=2)
            except Exception as e:
                return f"Retrieve failed: {e}"

        retrieve_button.click(on_retrieve, inputs=[lm_state, query, k], outputs=[retrieve_out])

        demo.launch(server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    start_app()
