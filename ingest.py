"""Document management CLI retained for compatibility with the original project."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from docwise.config import Settings
from docwise.exceptions import DocWiseError
from docwise.vector_store import VectorStoreManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index and manage DocWise documents")
    parser.add_argument("paths", nargs="*", type=Path, help="Files to index (defaults to data/)")
    parser.add_argument("--chunk-size", type=int, help="Characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, help="Overlapping characters")
    parser.add_argument("--force", action="store_true", help="Reindex duplicates")
    parser.add_argument("--list", action="store_true", help="List indexed documents")
    parser.add_argument("--delete", metavar="DOCUMENT_ID", help="Delete a document and its vectors")
    parser.add_argument("--reindex", metavar="DOCUMENT_ID", help="Reindex a stored document")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = Settings.from_env()
    manager = VectorStoreManager(settings)
    try:
        if args.list:
            records = manager.list_documents()
            if not records:
                print("No documents are indexed.")
            for record in records:
                print(
                    f"{record.document_id}  {record.filename}  "
                    f"{record.page_count} page(s), {record.chunk_count} chunk(s)"
                )
            return
        if args.delete:
            count = manager.delete_document(args.delete)
            print(f"Deleted document and {count} vector chunk(s).")
            return
        if args.reindex:
            result = manager.reindex_document(
                args.reindex,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            print(result.message)
            return

        paths = args.paths or [
            path
            for path in settings.data_dir.iterdir()
            if path.suffix.lower() in {".pdf", ".txt", ".md", ".markdown"}
        ]
        if not paths:
            raise SystemExit(f"Add PDF, TXT, or Markdown files to {settings.data_dir} first.")

        failures = 0
        for path in paths:
            try:
                result = manager.index_file(
                    path,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                    force=args.force,
                )
                print(f"[{result.status}] {result.message}")
            except (DocWiseError, OSError, ValueError) as exc:
                failures += 1
                print(f"[failed] {path.name}: {exc}")
        if failures:
            raise SystemExit(f"Indexing completed with {failures} failure(s).")
    except DocWiseError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
