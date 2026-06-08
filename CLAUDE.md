# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Recipe Recommender RAG system for a food startup. Answers natural-language user queries ("What can I make with leftover chicken and lemon?", "Give me a low-carb dessert under 30 minutes.") against a corpus of recipe documents using retrieval-augmented generation.

Full specification is in `resources/part1.txt`.

## Architecture

The system has three main layers:

**Ingestion Pipeline**
- Parse recipe text files (Title, Tags, Prep/Cook/Serves, Ingredients, Method, Chef Tips sections)
- Chunk per section — ingredients, method, and tips are always separate chunks; never merged
- Attach metadata to each chunk: `recipe_id`, `section`, `tags`, `prep_time_mins`, `cook_time_mins`, `serves`
- Embed chunks using a consistent model and store in a vector index

**Vector Index**
- Target: grow from 50K → 500K chunks over 12 months
- Index strategy: IVFFlat at small scale, migrate to HNSW at larger scale
- Embedding model and index parameters must be consistent between ingestion and retrieval

**Retrieval Function**
- Entry point: `retrieve_recipes(query: str, top_k: int = 5, nprobe: int = 10) -> list[dict]`
- Embed query with the same model used at ingestion
- Optional metadata pre-filter: parse dietary keywords from query (vegan, gluten-free, keto, low-carb) and filter by `tags` before vector search
- Return top-K chunks with chunk text + metadata

## Key Design Constraints

- Chunking granularity: one section per chunk (not one recipe, not one instruction step), so a single embedding carries enough signal for cosine similarity on ingredient + dietary filter queries
- Method sections can be 800+ words — no overlap needed since section boundaries are structural, not sliding-window
- The same embedding model must be used at ingestion time and query time; a hash-based dev shortcut must never reach production (guard with a model class check or env validation)
- Metadata pre-filtering should be measured for latency vs. recall tradeoffs against unfiltered search
