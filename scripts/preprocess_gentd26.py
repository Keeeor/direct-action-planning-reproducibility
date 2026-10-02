from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dap.data.gentd26 import preprocess_gentd26


UPSTREAM_BLOBS = {
    "README.md": "a2891c8fcd4b70357370004f33c5a00a72c67050",
    "qps.tar.gz": "794cdb0913bca65c59887be3ae39a9976afc36fc",
    "lora_request_trace.csv": "db8d3cfaea62e7086aff40e98ab6446619d9e5ce",
    "queue_size_raw_anon.tar.gz": "1165c5a1c794aa1a5f027d35613dca275bd9d223",
    "queue_rt_raw_anon.tar.gz": "92d635a794e2e9070e2b2a259ebcbc0208704cc8",
    "pod_gpu_duty_cycle_anon.tar.gz": "7e32b9794239e52d53d9f9eefd952ca95d0e5e19",
    "pod_gpu_memory_used_bytes_anon.tar.gz": "6758371e09cbe59623909d7643663467376d7f74",
    "pod_memory_util_anon.tar.gz": "d2e467a09f078423607fb8c470bfd4f1735a7833",
    "pipeline_inference_data_anon.tar.gz": "49e7cf19b3e967f1cf2193a473eab42e412d2a55",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / "data/raw/gentd26"
    processed = root / "data/processed/gentd26"
    metadata_path = root / "data/metadata/gentd26/dataset.json"
    metadata = preprocess_gentd26(raw, processed)
    metadata["raw_sha256"] = {
        name: sha256(raw / name) for name in UPSTREAM_BLOBS
    }
    metadata["upstream_git_blob_sha1"] = UPSTREAM_BLOBS
    metadata["processed_sha256"] = {
        path.name: sha256(path) for path in sorted(processed.glob("*.npz"))
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(metadata_path)


if __name__ == "__main__":
    main()
