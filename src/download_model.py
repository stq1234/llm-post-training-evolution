"""Download the pinned Qwen2.5-0.5B-Instruct files to the HF default cache."""

from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def download_model() -> Path:
    """Fetch only model and tokenizer files at the immutable revision."""
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            allow_patterns=ALLOW_PATTERNS,
            token=False,
        )
    )
    if snapshot.name != REVISION:
        raise RuntimeError(
            f"Unexpected cached revision directory: {snapshot.name!r}"
        )

    missing = [name for name in ALLOW_PATTERNS if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Downloaded snapshot is missing: {missing}")
    return snapshot


def main() -> None:
    snapshot = download_model()
    print(f"model_id={MODEL_ID}")
    print(f"revision={REVISION}")
    print(f"snapshot={snapshot}")
    for name in ALLOW_PATTERNS:
        file = snapshot / name
        print(f"{name}\t{file.stat().st_size} bytes")


if __name__ == "__main__":
    main()
