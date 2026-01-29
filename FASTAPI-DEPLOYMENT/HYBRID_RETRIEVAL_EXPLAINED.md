# Hybrid Retrieval Pipeline - Detailed Breakdown

## Overview
The `hybrid_retrieve()` function implements a **multi-stage retrieval system** that combines vector similarity, deduplication, and re-ranking to select the most relevant chunks.

```
User Query
    ↓
┌─────────────────────────────────────────────────────┐
│ STAGE 1: VECTOR SEARCH (Semantic Matching)          │
│ - Encode query → Find similar chunks by meaning     │
│ - Returns: Top K semantically similar chunks        │
└─────────────────────────────────────────────────────┘
    ↓ (Vector Matches)
┌─────────────────────────────────────────────────────┐
│ STAGE 2: PROCESS & DEDUPLICATE                      │
│ - Clean chunks, remove duplicates                   │
│ - Cap to U_CAP limit                                │
└─────────────────────────────────────────────────────┘
    ↓ (Cleaned & Capped Candidates)
┌─────────────────────────────────────────────────────┐
│ STAGE 3: RE-RANKING (Cross-Encoder Scoring)        │
│ - Score each chunk against query using BERT         │
│ - Relevance score indicates how relevant chunk is   │
└─────────────────────────────────────────────────────┘
    ↓ (Scored Candidates)
┌─────────────────────────────────────────────────────┐
│ STAGE 4: SORT BY RELEVANCE                          │
│ - Sort by cross-encoder score (highest first)       │
│ - Final ranked order                                │
└─────────────────────────────────────────────────────┘
    ↓
FINAL OUTPUT: Ranked candidates [chunk0, chunk1, ..., chunkN]
    ↓
SPLIT FOR ANSWER & FOLLOW-UP:
    Answer chunks:   chunks[0:4]   (top 4)
    FollowUp chunks: chunks[4:6]   (next 2)
```

---

## Stage 1: Vector Search (Semantic Matching)

### Input
- `query: str` - User's medical question
- `top_k_vec: int = 10` - How many chunks to retrieve from vector DB

### Process
```python
# Step 1a: Encode query into embedding
q_emb = embedding_model.encode(query, convert_to_numpy=True).tolist()
# Example: "What is jaundice?" → [0.234, -0.156, 0.789, ..., 0.045] (384-dim vector)

# Step 1b: Query ChromaDB for similar chunks
results = chromadb_collection.query(
    query_embeddings=[q_emb],
    n_results=min(top_k_vec, collection_count),  # Request up to 10 chunks
    include=["metadatas", "documents", "distances"]
)
```

### Output Structure (Example with 3 results)
```
ChromaDB returns:
{
  "ids": [[chunk_001, chunk_042, chunk_156]],
  "documents": [[
    "Jaundice is yellowing of skin due to bilirubin...",
    "Newborn jaundice causes: immature liver function...",
    "Treatment of jaundice: phototherapy, exchange transfusion..."
  ]],
  "metadatas": [[
    {"source": "enc1.pdf", "page": 5, "text_full": "..."},
    {"source": "enc1.pdf", "page": 6, "text_full": "..."},
    {"source": "pee.pdf", "page": 2, "text_full": "..."}
  ]],
  "distances": [[0.15, 0.23, 0.31]]  # Lower = more similar
}
```

### How Similarity Works
```
ChromaDB uses Cosine Distance:
  distance = 1 - cosine_similarity
  
  Example:
  Query: "What is jaundice?"
  Chunk A: "Jaundice is yellowing..."     → distance=0.10 → similarity=0.90 ✅ (very similar)
  Chunk B: "Fever causes..."              → distance=0.70 → similarity=0.30 ❌ (not similar)
  
  Conversion formula in code:
  similarity = max(0.0, 1.0 - distance)
```

### Result After Stage 1
```
vec_matches = [
  {
    "id": "chunk_001",
    "text": "Jaundice is yellowing of skin due to bilirubin...",
    "meta": {"source": "enc1.pdf", "page": 5}
  },
  {
    "id": "chunk_042",
    "text": "Newborn jaundice causes: immature liver function...",
    "meta": {"source": "enc1.pdf", "page": 6}
  },
  {
    "id": "chunk_156",
    "text": "Treatment of jaundice: phototherapy...",
    "meta": {"source": "pee.pdf", "page": 2}
  }
]
# Total: 10 chunks (if available)
```

---

## Stage 2: Process & Deduplicate + Cap

### Purpose
- Remove invalid/empty chunks
- Eliminate duplicate IDs
- Limit to maximum `u_cap` chunks

### Parameters
- `u_cap: int = 7` - Maximum chunks to keep (union cap)

### Process
```python
# 2a: Deduplicate by ID
candidates = {m["id"]: m for m in vec_matches}  # Dict removes duplicates automatically
# If 2 chunks have same ID, only 1 is kept

# 2b: Cap to u_cap limit
candidates = list(candidates.values())[:u_cap]  # Keep only first 7
```

### Input/Output Flow
```
Input:  10 chunks from vector search
  ↓
Deduplicate: 10 chunks (assume no duplicates)
  ↓
Cap to 7: Keep only top 7
  ↓
Output: 7 candidates
```

### Candidates Structure After Stage 2
```
candidates = [
  {
    "id": "chunk_001",
    "text": "Jaundice is yellowing of skin...",
    "meta": {"source": "enc1.pdf", "page": 5}
  },
  {
    "id": "chunk_042",
    "text": "Newborn jaundice causes: immature liver...",
    "meta": {"source": "enc1.pdf", "page": 6}
  },
  # ... up to 7 total
]
```

---

## Stage 3: Re-ranking with Cross-Encoder (BERT-based)

### Purpose
**Score each chunk against the original query using a BERT model**

The vector search finds *semantically similar* chunks, but a cross-encoder provides a *relevance score* that's more nuanced.

### Two Implementation Options

#### Option A: FlashRank (Current Preference)
```python
if hasattr(reranker, "rerank"):
    # FlashRank is optimized ONNX-based reranker
    passages = [{"id": c["id"], "text": c["text"], "meta": c["meta"]} for c in candidates]
    request = RerankRequest(query=query, passages=passages)
    ranked = reranker.rerank(request)
    
    # Extract scores from ranked results
    for c in candidates:
        matching = next((r for r in ranked if r.get("id") == c.get("id")), None)
        c["scores"] = {"cross": float(matching.get("relevance_score", 0.0))}
```

#### Option B: CrossEncoder (Fallback)
```python
else:
    # CrossEncoder is slower but more accurate
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    
    for i, c in enumerate(candidates):
        c["scores"] = {"cross": float(scores[i])}
```

### How Cross-Encoder Scoring Works

**Example: Query "What is jaundice?" with 3 candidates**

```
Query: "What is jaundice?"

Candidate 1 (chunk_001):
  Text: "Jaundice is yellowing of skin and eyes due to bilirubin accumulation"
  Cross-Encoder Input: [query, text]
  Score: 0.85 ✅ (Very relevant - directly answers question)

Candidate 2 (chunk_042):
  Text: "Causes of newborn jaundice include immature liver function"
  Cross-Encoder Input: [query, text]
  Score: 0.72 ✅ (Relevant - related to jaundice)

Candidate 3 (chunk_156):
  Text: "Fever can cause confusion and weakness in patients"
  Cross-Encoder Input: [query, text]
  Score: 0.15 ❌ (Not relevant - about fever, not jaundice)
```

### Candidates After Stage 3 (With Scores)
```
candidates = [
  {
    "id": "chunk_001",
    "text": "Jaundice is yellowing of skin...",
    "meta": {"source": "enc1.pdf"},
    "scores": {"cross": 0.85}  # ← NEW: Relevance score added
  },
  {
    "id": "chunk_042",
    "text": "Newborn jaundice causes...",
    "meta": {"source": "enc1.pdf"},
    "scores": {"cross": 0.72}
  },
  {
    "id": "chunk_156",
    "text": "Fever can cause confusion...",
    "meta": {"source": "pee.pdf"},
    "scores": {"cross": 0.15}
  }
]
```

---

## Stage 4: Sort by Relevance Score

### Process
```python
candidates = sorted(
    candidates,
    key=lambda x: x.get("scores", {}).get("cross", 0.0),
    reverse=True  # Highest score first
)
```

### Before Sorting
```
candidates = [
  {"id": "chunk_001", "scores": {"cross": 0.85}},  # Position 0
  {"id": "chunk_042", "scores": {"cross": 0.72}},  # Position 1
  {"id": "chunk_156", "scores": {"cross": 0.15}},  # Position 2
]
```

### After Sorting (Highest Score First)
```
candidates = [
  {"id": "chunk_001", "scores": {"cross": 0.85}},  # Position 0 ← Best
  {"id": "chunk_042", "scores": {"cross": 0.72}},  # Position 1
  {"id": "chunk_156", "scores": {"cross": 0.15}},  # Position 2 ← Worst
]
```

---

## Final Output: Ranked Candidates Ready for Split

```python
# Returned from hybrid_retrieve()
candidates = [
  # INDEX 0
  {
    "id": "chunk_001",
    "text": "Jaundice is yellowing of skin and eyes...",
    "meta": {"source": "enc1.pdf", "page": 5},
    "scores": {"cross": 0.85}
  },
  # INDEX 1
  {
    "id": "chunk_042",
    "text": "Newborn jaundice causes: immature liver...",
    "meta": {"source": "enc1.pdf", "page": 6},
    "scores": {"cross": 0.72}
  },
  # INDEX 2
  {
    "id": "chunk_156",
    "text": "Treatment of jaundice: phototherapy...",
    "meta": {"source": "pee.pdf", "page": 2},
    "scores": {"cross": 0.68}
  },
  # INDEX 3
  {
    "id": "chunk_201",
    "text": "Prevention of jaundice through...",
    "meta": {"source": "enc2.pdf", "page": 3},
    "scores": {"cross": 0.62}
  },
  # INDEX 4
  {
    "id": "chunk_305",
    "text": "Complications of untreated jaundice...",
    "meta": {"source": "enc1.pdf", "page": 8},
    "scores": {"cross": 0.58}
  },
  # INDEX 5
  {
    "id": "chunk_410",
    "text": "Long-term effects of neonatal jaundice...",
    "meta": {"source": "pee.pdf", "page": 4},
    "scores": {"cross": 0.52}
  },
]
```

---

## Splitting for Answer vs Follow-up

### Current Behavior (All for Answer)
```python
# ALL 6 chunks go to answer generation
answer_candidates = candidates  # [0, 1, 2, 3, 4, 5]
```

### NEW Behavior (After Follow-up Feature)
```python
# Split: 4 for answer, 2 for follow-up
answer_candidates = candidates[:4]   # [0, 1, 2, 3]
followup_candidates = candidates[4:6]  # [4, 5]

# Answer chunks:
# [0]: "Jaundice is yellowing..." (0.85)
# [1]: "Causes: immature liver..." (0.72)
# [2]: "Treatment: phototherapy..." (0.68)
# [3]: "Prevention through..." (0.62)

# Follow-up chunks:
# [4]: "Complications of untreated..." (0.58)
# [5]: "Long-term effects..." (0.52)
```

---

## Edge Case Handling

### Edge Case 1: Fewer than 6 Chunks Retrieved
```
Scenario: hybrid_retrieve() returns only 4 chunks

Candidates: [chunk_001, chunk_042, chunk_156, chunk_201]

Splitting Logic:
  answer_count = min(4, len(candidates)) = 4
  followup_count = max(0, len(candidates) - 4) = 0
  
  answer_candidates = candidates[:4]      # [0, 1, 2, 3] - ALL 4
  followup_candidates = candidates[4:6]   # [] - EMPTY
  
Result: No follow-up generated
```

### Edge Case 2: Only 3 Chunks Retrieved
```
Scenario: hybrid_retrieve() returns 3 chunks (very rare)

Candidates: [chunk_001, chunk_042, chunk_156]

answer_candidates = candidates[:4]      # [0, 1, 2] - ALL 3
followup_candidates = candidates[4:6]   # [] - EMPTY

Result: Use all 3 for answer, no follow-up
```

### Edge Case 3: Empty Results (0 chunks)
```
Scenario: No relevant chunks found

Candidates: []

answer_candidates = candidates[:4]      # [] - EMPTY
followup_candidates = candidates[4:6]   # [] - EMPTY

Result: 
  - Intent = "no_context"
  - Answer: "Sorry, I don't have information about this"
  - Follow-up: Optional - suggest related topics
```

---

## Component Summary Table

| Component | Stage | Input | Process | Output |
|-----------|-------|-------|---------|--------|
| **Query Encoder** | 1 | Question text | SentenceTransformer | 384-dim vector |
| **ChromaDB** | 1 | Query vector | Find similar embeddings | Top 10 chunks |
| **Deduplicator** | 2 | 10 chunks | Remove duplicate IDs | ~10 chunks (unique) |
| **Capper** | 2 | ~10 chunks | Limit to u_cap=7 | 7 chunks max |
| **CrossEncoder/FlashRank** | 3 | Query + text pairs | Score relevance | 7 chunks with scores |
| **Sorter** | 4 | Scored chunks | Sort by score desc | 7 ranked chunks |
| **Splitter** | 5 | 7 ranked chunks | Split at index 4 | 4 answer + 3 follow-up (or fewer) |

---

## Data Flow Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│ User Query: "What is jaundice?"                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
        ┌──────────────────────────────────────┐
        │ STAGE 1: Vector Search               │
        │ embedding_model.encode() → ChromaDB  │
        └──────────────────────────┬───────────┘
                                   ↓
                        10 chunks (vector order)
                        [0.90, 0.88, 0.85, 0.82, 0.78, 0.75, 0.68, 0.65, 0.60, 0.55]
                                   ↓
        ┌──────────────────────────────────────┐
        │ STAGE 2: Process & Deduplicate & Cap │
        └──────────────────────────┬───────────┘
                                   ↓
                        7 chunks (deduplicated, capped)
                                   ↓
        ┌──────────────────────────────────────┐
        │ STAGE 3: Cross-Encoder Re-ranking    │
        │ FlashRank/CrossEncoder.predict()     │
        └──────────────────────────┬───────────┘
                                   ↓
                    7 chunks with cross scores
                    [0.85, 0.72, 0.68, 0.62, 0.58, 0.52, 0.48]
                                   ↓
        ┌──────────────────────────────────────┐
        │ STAGE 4: Sort by Relevance           │
        │ sorted(by cross score, desc=True)    │
        └──────────────────────────┬───────────┘
                                   ↓
                    FINAL: 7 ranked candidates
                    [0.85, 0.72, 0.68, 0.62, 0.58, 0.52, 0.48]
                                   ↓
        ┌──────────────────────────────────────┐
        │ STAGE 5: SPLIT                       │
        │ Answer:   [0:4] = 4 chunks           │
        │ FollowUp: [4:6] = 2 chunks (if avail)│
        └──────────────────────────┬───────────┘
                                   ↓
                    ✅ Ready for pipeline
```

---

## Key Insights

1. **Two-Pass Scoring**: Vector search finds candidates, then cross-encoder refines ranking
2. **Semantic ≠ Relevant**: A chunk similar to query might not directly answer it
3. **Consistent Order**: Always sorted by relevance score (highest first)
4. **Deterministic Split**: Answer chunks = [0:4], Follow-up = [4:6] (or fewer)
5. **Fallback Support**: If < 4 chunks, all go to answer; no follow-up
6. **No Blocking**: All stages are fast (total ~1-2s for entire pipeline)
