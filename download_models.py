"""下载 bge 系列模型到本地 models/ 目录（走 ModelScope 国内镜像，避免 HuggingFace 直连慢）。

用法：python download_models.py
"""
from modelscope import snapshot_download

MODELS = [
    ("AI-ModelScope/bge-large-zh-v1.5", "models/bge-large-zh-v1.5"),
    ("AI-ModelScope/bge-reranker-base", "models/bge-reranker-base"),
]


def main() -> None:
    for model_id, local_dir in MODELS:
        print(f"[downloading] {model_id} -> {local_dir}", flush=True)
        path = snapshot_download(model_id, local_dir=local_dir)
        print(f"[done] {path}", flush=True)


if __name__ == "__main__":
    main()
