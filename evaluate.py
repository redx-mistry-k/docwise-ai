"""Run labeled retrieval evaluation from the command line."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

from docwise.config import Settings
from docwise.evaluation import benchmark_retrieval_strategies, load_dataset
from docwise.retrieval import Retriever
from docwise.vector_store import VectorStoreManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DocWise retrieval")
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/example_questions.json"))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=350)
    parser.add_argument("--chunk-overlap", type=int, default=60)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        help="Index this corpus into an isolated temporary store before benchmarking.",
    )
    parser.add_argument("--output", type=Path, help="Optionally save the JSON report.")
    args = parser.parse_args()
    settings = Settings.from_env()
    examples = load_dataset(args.dataset)

    def run(active_settings: Settings) -> dict:
        manager = VectorStoreManager(active_settings)
        if args.corpus_dir:
            manager.index_directory(
                args.corpus_dir,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        return benchmark_retrieval_strategies(
            Retriever(manager),
            examples,
            top_k=args.top_k,
        )

    if args.corpus_dir:
        settings.storage_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="benchmark-",
            dir=settings.storage_dir,
            ignore_cleanup_errors=True,
        ) as temporary:
            storage = Path(temporary)
            benchmark_settings = replace(
                settings,
                storage_dir=storage,
                chroma_dir=storage / "chroma",
                registry_path=storage / "registry.json",
                usage_db_path=storage / "gemini_usage.sqlite3",
            )
            result = run(benchmark_settings)
    else:
        result = run(settings)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
