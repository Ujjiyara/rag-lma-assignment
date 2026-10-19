# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking

Two retrieval pipelines over the same chunked corpus, the **CATAN Game Rules & Almanac** (5th ed., 16 pages):

| | Pipeline 1 (baseline) | Pipeline 2 (re-ranking) |
|---|---|---|
| Stage 1 | Bi-encoder `all-MiniLM-L6-v2`, cosine similarity, Top-10 | same Top-10 |
| Stage 2 | — | Cross-encoder `ms-marco-MiniLM-L-6-v2` re-scores the 10 pairs |
| LLM context | Top-3 by cosine | Top-3 by cross-encoder score |
| Generator | Gemini (`gemini-flash-lite-latest` in the saved run; the notebook falls back through several Flash models when one is overloaded or out of quota, always using the same model for both answers), identical prompt, temperature 0 | same |

Everything is in **`rag_baseline_vs_rerank.ipynb`**: PDF extraction and cleaning, chunking, both pipelines,
failure-query rank shifts, Answer 1 vs. Answer 2, a wider 29-question retrieval evaluation, latency profiling
and a scaling measurement.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env          # then put your Gemini API key in .env
jupyter notebook rag_baseline_vs_rerank.ipynb
```

The notebook downloads the rulebook PDF from catan.com into `data/` on first run and runs on CPU.
It needs 8 Gemini answers (4 failure queries × 2 pipelines). All answers from the saved run are cached in
`llm_cache.json`, so re-running reproduces the same results without an API key or any quota.

## Files

- `rag_baseline_vs_rerank.ipynb`: all code, with outputs
- `REPORT.md`: the report
- `llm_cache.json`: raw Gemini answers and their latencies from the saved run
