"""单独下载 bge-reranker-base（只下 sentence_transformers 需要的文件，跳过重复的 onnx/pytorch 权重）。"""
from modelscope import snapshot_download

if __name__ == "__main__":
    print("[downloading] BAAI/bge-reranker-base -> models/bge-reranker-base", flush=True)
    path = snapshot_download(
        "BAAI/bge-reranker-base",
        local_dir="models/bge-reranker-base",
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
        ],
    )
    print("[done]", path, flush=True)
