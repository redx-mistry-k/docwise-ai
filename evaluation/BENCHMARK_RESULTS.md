# DocWise Retrieval Benchmark Results

Measured locally on 2026-10-09 with Python 3.12, the bundled 10-document synthetic corpus, 41 manually labeled questions, 350-character chunks, 60-character overlap, cached `all-MiniLM-L6-v2` embeddings, and no Gemini calls.

Four missing-information questions are excluded from retrieval Recall/Precision/MRR because they intentionally have no relevant source. They remain part of answer-refusal evaluation, leaving 37 retrieval-scored questions.

## Top-k 2

| Strategy | Recall@K | Precision@K | MRR | Mean latency |
|---|---:|---:|---:|---:|
| Baseline vector | 0.86 | 0.49 | 0.88 | 10.20 ms |
| BM25 | 0.85 | 0.49 | 0.91 | 0.24 ms |
| Hybrid | 0.86 | 0.50 | 0.92 | 10.38 ms |
| Hybrid + reranking | 0.86 | 0.50 | 0.92 | 10.23 ms |
| Intelligent adaptive retrieval | 0.88 | 0.50 | 0.92 | 11.51 ms |

At top-k 2, adaptive retrieval improved aggregate recall by 0.02 over hybrid reranking. Precision and MRR were unchanged, while mean retrieval latency increased by 1.28 ms. No benchmark question triggered the bounded retry, so this result measures decomposition/adaptive depth rather than retry recovery.

## Top-k 4

| Strategy | Recall@K | Precision@K | MRR | Mean latency |
|---|---:|---:|---:|---:|
| Baseline vector | 0.93 | 0.28 | 0.89 | 11.11 ms |
| BM25 | 0.92 | 0.28 | 0.91 | 0.24 ms |
| Hybrid | 0.93 | 0.28 | 0.93 | 10.80 ms |
| Hybrid + reranking | 0.93 | 0.28 | 0.93 | 10.94 ms |
| Intelligent adaptive retrieval | 0.93 | 0.28 | 0.93 | 12.25 ms |

At top-k 4, the intelligent pipeline did not improve retrieval quality on this corpus and added about 1.31 ms over hybrid reranking. The bounded-retry path is verified separately with deterministic automated tests covering both successful recovery and final refusal.

Latency is machine-specific and includes local retrieval only. These figures are not Gemini generation latency and should not be generalized to production-sized corpora.

## Reproduce

```powershell
$env:HF_HUB_OFFLINE='1' # optional when the embedding model is already cached
$env:TRANSFORMERS_OFFLINE='1'
python evaluate.py `
  --dataset evaluation\docwise_benchmark_40.json `
  --corpus-dir evaluation\benchmark_corpus `
  --top-k 2 `
  --chunk-size 350 `
  --chunk-overlap 60 `
  --output .docwise\benchmark_report_top2.json
```
