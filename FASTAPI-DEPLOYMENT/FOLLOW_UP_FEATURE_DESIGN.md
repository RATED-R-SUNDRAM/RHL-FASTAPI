# Follow-Up Question Feature Design Document

## Feature Overview
After generating a medical answer, suggest contextual follow-up questions based on retrieved chunks. This extends the conversation naturally without impacting response latency.

---

## Architecture Design

### Phase 1: Retrieval Split (Hybrid Retrieval)
```
hybrid_retrieve() → returns sorted candidates

Current: All top-N chunks used for answer framing
New:     Split into:
         - Answer Chunks: Top 4 candidates (indexes 0-3)
         - FollowUp Chunks: Next 2 candidates (indexes 4-5)
         
Example:
  Chunk 0: Jaundice definition
  Chunk 1: Jaundice symptoms
  Chunk 2: Jaundice causes
  Chunk 3: Jaundice diagnosis
  ---|Answer Generation---|
  Chunk 4: Jaundice treatment options ← Follow-up candidate
  Chunk 5: Prevention strategies ← Follow-up candidate
```

### Phase 2: Topic Extraction (LLM-based)
```
Input:  Answer text + 2 Follow-up chunks
Output: 
  {
    "answer_topics": {
      "main_topic": "jaundice",
      "sub_topics": ["definition", "symptoms", "causes"]
    },
    "followup_topics": {
      "chunk_4": {
        "main_topic": "jaundice",
        "sub_topics": ["treatment", "management"],
        "novel_content": true
      },
      "chunk_5": {
        "main_topic": "jaundice",
        "sub_topics": ["prevention", "lifestyle"],
        "novel_content": true
      }
    }
  }
```

### Phase 3: Follow-up Question Generation
```
Logic:
  For each follow-up chunk:
    IF chunk.main_topic == answer.main_topic AND
       chunk.novel_sub_topics NOT IN answer.sub_topics:
      → Generate: "Would you like to know about <sub_topic> of <main_topic>?"
    ELSE:
      → Skip this chunk

Examples:
  ✅ Answer covers: "What is jaundice?" (definition only)
     Follow-up chunk covers: "Treatment of jaundice"
     → "Would you like to know about treatment of jaundice?"
     
  ❌ Answer covers: "Jaundice: definition, symptoms, causes"
     Follow-up chunk covers: "Symptoms of jaundice"
     → SKIP (already covered in answer)
     
  ❌ Answer covers: "Jaundice causes"
     Follow-up chunk covers: "Fever causes"
     → SKIP (different main topic)
```

---

## Edge Cases & Solutions

### Edge Case 1: Fewer than 6 Chunks Retrieved
```
Problem: hybrid_retrieve() returns < 6 chunks

Solutions:
  • If 5 chunks: Use 4 for answer, 1 for follow-up
  • If 4 chunks: Use all 4 for answer, NO follow-up
  • If 3 chunks: Use all 3 for answer, NO follow-up
  • If < 3 chunks: Use all for answer, NO follow-up
  
Logic:
  answer_chunks = min(4, len(candidates))
  followup_chunks = max(0, len(candidates) - answer_chunks)
  
  if followup_chunks == 0:
    follow_up_question = None
```

### Edge Case 2: No Follow-up Content After Filtering
```
Problem: Follow-up chunks exist but don't have novel content

Example:
  Answer: "Jaundice: definition, symptoms, causes, treatment, prevention"
  FollowUp Chunk 4: "More about jaundice symptoms"
  FollowUp Chunk 5: "More about jaundice treatment"
  
Both have content already in answer → No follow-up generated

Behavior: follow_up_question = None
          (Don't force a follow-up if there's nothing new to offer)
```

### Edge Case 3: Follow-up for "No Context" Responses
```
Problem: When answer is "I don't have information", should there be a follow-up?

Solutions:
  • Option A: NO follow-up (answer is already "sorry")
  • Option B: SMART follow-up - suggest related topic
    Example:
      User asks: "Can you tell me about alien pregnancy?"
      Answer: "Sorry, I don't have info about this"
      FollowUp: "Would you like to know about normal pregnancy care?"
  
Recommended: Option B (helps redirect to relevant topics)

Logic:
  if intent == "no_context" AND followup_chunks exist:
    → Generate generic follow-up from follow-up chunks
    → "Would you like to know about <alternative_topic>?"
```

### Edge Case 4: Only 1 Follow-up Chunk Available
```
Problem: Only 1 chunk qualifies for follow-up

Behavior:
  • Use that 1 chunk for follow-up
  • Generate single follow-up question
  
Logic:
  followup_chunks = [chunk_5]  # Only 1 chunk
  if followup_chunks[0].has_novel_content():
    follow_up = generate_followup_question(followup_chunks[0])
  else:
    follow_up = None
```

### Edge Case 5: Topic Mismatch Between Chunks
```
Problem: Follow-up chunk is about completely different topic

Example:
  Answer: "Jaundice: definition and symptoms"
  FollowUp Chunk: "Fever: causes and treatment"
  
Behavior:
  → Topic mismatch (jaundice != fever)
  → NO follow-up generated
  → follow_up_question = None
```

---

## Time to First Token Constraint ⏱️

### Requirement
**Time to first token MUST NOT increase due to follow-up generation**

### Current Flow
```
User Request → LLM Processing → First Token → Stream Rest → Metadata
                                    ↑
                              (User gets response)
```

### Solution: Asynchronous Follow-up Generation
```
User Request → LLM Processing → First Token → Stream Rest → Metadata (+ follow-up)
                                    ↑              ↓
                              Stream starts   Compute follow-up
                                             in parallel
                                             (non-blocking)
```

### Implementation Strategy
1. **During initial answer generation**: Use top 4 chunks only
2. **After first token is yielded**: Launch async task to compute follow-up from chunks 4-5
3. **Add follow-up to final metadata**: Include in the final JSON metadata sent to client

### Pseudo-code
```python
async def medical_pipeline_api_stream(...):
    # ... existing code ...
    
    # Retrieve all candidates (could be 6+)
    all_candidates = hybrid_retrieve(rewritten)  # e.g., [0,1,2,3,4,5,6,7,...]
    
    # Split: 4 for answer, rest for follow-up
    answer_candidates = all_candidates[:4]
    followup_candidates = all_candidates[4:6]  # Take only next 2
    
    # Generate answer from answer_candidates (no change)
    answer = generate_answer(answer_candidates)
    
    # Stream first token immediately
    yield first_token
    time_to_first_token = now() - start
    
    # Launch async task for follow-up (non-blocking)
    # This happens while other tokens are being streamed
    followup_task = asyncio.create_task(
        extract_and_generate_followup(answer, followup_candidates)
    )
    
    # Stream remaining tokens
    for token in remaining_tokens:
        yield token
    
    # Wait for follow-up task to complete (should be ready by now)
    followup_question = await followup_task
    
    # Add to metadata
    metadata = {
        'answer': answer,
        'follow_up': followup_question,  # New field
        'time_to_first_token': time_to_first_token
    }
```

---

## Follow-up Question Prompts

### Prompt 1: Topic & Sub-topic Extraction
```
Purpose: Extract main topic and sub-topics from answer and follow-up chunks

Template:
"
Return JSON only: 
{
  "answer_analysis": {
    "main_topic": "medical condition or procedure name",
    "sub_topics": ["aspect1", "aspect2", "aspect3"],
    "coverage_summary": "what aspects of main_topic are covered"
  },
  "followup_analysis": [
    {
      "chunk_id": 4,
      "main_topic": "medical condition or procedure name",
      "sub_topics": ["aspect1", "aspect2"],
      "novel_aspects": ["aspect not in answer"],
      "relevance": "strong|weak|unrelated"
    },
    {
      "chunk_id": 5,
      "main_topic": "medical condition or procedure name",
      "sub_topics": ["aspect1", "aspect2"],
      "novel_aspects": ["aspect not in answer"],
      "relevance": "strong|weak|unrelated"
    }
  ],
  "recommendation": {
    "has_followup": true|false,
    "suggested_chunk": 4 or 5,
    "followup_aspect": "specific sub-topic to ask about"
  }
}

RULES:
- Identify ONE main medical topic from each chunk
- List ALL unique sub-topics/aspects covered
- Assess which aspects are NEW (not in answer)
- Rate relevance: 
  * strong: Same main topic, novel sub-topics
  * weak: Related but different topic
  * unrelated: Completely different topic
- Prefer novelty: Choose chunk with most novel, relevant aspects

Examples:
- Answer: "Jaundice definition and symptoms"
  Chunk 4: "Jaundice treatment options"
  → recommendation.has_followup = true
  → suggested_chunk = 4
  → followup_aspect = "treatment"

- Answer: "Jaundice: definition, symptoms, causes, treatment"
  Chunk 4: "Jaundice symptoms (more details)"
  → recommendation.has_followup = false (already covered)

- Answer: "Jaundice definition"
  Chunk 4: "Fever causes"
  → recommendation.has_followup = false (unrelated)
"
```

### Prompt 2: Follow-up Question Generation
```
Purpose: Generate natural follow-up questions

Template:
"
Generate a follow-up question based on the provided information.

Context:
- Current answer covers: <topics covered in answer>
- Additional information available in documents: <novel topics in chunk>
- Main medical topic: <main_topic>

Rules:
1. Format: Always use 'Would you like to know about <specific topic>?'
2. Specificity: Be precise about what aspect is being offered
3. Naturalness: Sound like a natural conversation flow
4. Relevance: Clearly link to the main topic discussed
5. Variety: Use different phrasings for different scenarios

Examples:
  ✅ "Would you like to know about treatment options for jaundice?"
  ✅ "Would you like to know about prevention of newborn jaundice?"
  ✅ "Would you like to know about when to seek medical help for jaundice?"
  ✅ "Would you like to know about complications if jaundice is not treated?"
  
  ❌ "Would you like more info?" (too vague)
  ❌ "Do you want to know about treatment?" (not specific enough)
  ❌ "We also have information about other topics." (not offered)

Generate exactly ONE follow-up question following the rules above.
"
```

### Prompt 3: Follow-up for No-Context Responses
```
Purpose: Suggest relevant topics when no direct answer found

Template:
"
The assistant couldn't find information about the user's specific query.
Suggest an alternative, related medical topic that IS available in documents.

User asked about: <original_query>
Available topics in follow-up documents: <extracted topics from chunks>

Rules:
1. Suggest a RELATED topic (not random)
2. Format: 'Would you like to know about <alternative_topic> instead?'
3. Be helpful and relevant
4. Acknowledge the limitation gracefully

Example:
- User: "Tell me about alien pregnancy complications"
- Available: [jaundice, newborn care, breastfeeding]
- Response: "I don't have information about that, but would you like to know about common newborn complications like jaundice?"

Generate ONE helpful follow-up suggestion.
"
```

---

## Prompt Comprehensiveness Checklist

### Must Extract
- [ ] Main medical topic from each chunk
- [ ] All sub-topics/aspects covered (not just first mention)
- [ ] Novel aspects not in answer
- [ ] Relationship between answer topics and follow-up topics
- [ ] Confidence level in relevance

### Examples to Include
- [ ] Same main topic with novel sub-topic ✅
- [ ] Same main topic, overlapping content ❌
- [ ] Different but related topics ❌
- [ ] Completely unrelated topics ❌
- [ ] Complex multi-aspect topics
- [ ] Single-aspect topics

### Edge Cases Handled
- [ ] No follow-up chunks (< 4 total)
- [ ] All follow-up chunks already covered
- [ ] Follow-up chunks from completely different topics
- [ ] Ambiguous topic boundaries
- [ ] When to suppress follow-up

---

## Integration Points

### 1. Hybrid Retrieval Modification
```python
def hybrid_retrieve(...) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns:
      - answer_candidates: Top 4 chunks
      - followup_candidates: Chunks 4-5 (or fewer if <6 total)
    """
    candidates = [... existing retrieval logic ...]
    
    answer_candidates = candidates[:4]
    followup_candidates = candidates[4:6]
    
    return answer_candidates, followup_candidates
```

### 2. LLM Pipeline
```python
async def extract_followup_info(answer_text, followup_chunks) -> Dict:
    """Extract topics and generate follow-up question"""
    # Call Gemini LLM with topic extraction prompt
    # Parse JSON response
    # Return structured follow-up data
```

### 3. Streaming Response
```python
async def medical_pipeline_api_stream(...):
    # Existing code...
    
    # Launch async follow-up generation (after first token)
    followup_task = asyncio.create_task(
        extract_followup_info(answer, followup_chunks)
    )
    
    # Stream answer tokens
    # ...
    
    # Get follow-up result
    followup_result = await followup_task
    
    # Add to metadata
    metadata['follow_up'] = followup_result.get('question')
```

---

## Response Format

### Current
```json
{
  "type": "metadata",
  "answer": "Jaundice is...",
  "intent": "answer",
  "video_url": null,
  "time_to_first_token": 2.5,
  "total_time": 3.1
}
```

### New
```json
{
  "type": "metadata",
  "answer": "Jaundice is...",
  "intent": "answer",
  "video_url": null,
  "follow_up": "Would you like to know about treatment of jaundice?",
  "time_to_first_token": 2.5,
  "total_time": 3.1
}
```

### When No Follow-up
```json
{
  "type": "metadata",
  "answer": "Jaundice is...",
  "intent": "answer",
  "video_url": null,
  "follow_up": null,
  "time_to_first_token": 2.5,
  "total_time": 3.1
}
```

---

## Testing Strategy

### Test Cases
1. **Happy Path**: 6+ chunks → 4 for answer, 2 for follow-up → Novel content found → Follow-up generated
2. **Edge: <6 chunks**: 5 chunks → 4 for answer, 1 for follow-up
3. **Edge: No novel content**: 6 chunks → 4 for answer, 2 for follow-up → No novel content → follow_up = None
4. **Edge: No context**: Intent = "no_context" → Generate alternative topic follow-up
5. **Time constraint**: Verify time_to_first_token unchanged
6. **Async validation**: Verify follow-up task doesn't block streaming

### Timing Benchmarks
```
Before: 
  First token: 2.5s
  Total: 3.1s

After:
  First token: 2.5s (unchanged ✅)
  Total: 3.2s (small increase due to follow-up generation in parallel)
```

---

## Summary: 5 Key Constraints

| # | Constraint | Solution |
|---|-----------|----------|
| 1 | Time to first token unchanged | Async follow-up generation after first token |
| 2 | Hybrid edge cases (<6 chunks) | Smart splitting: 4 for answer, rest for follow-up |
| 3 | Comprehensive prompts | Extract topics, sub-topics, novelty, relevance |
| 4 | Follow-up not in all answers | Conditional: only if novel content found |
| 5 | For no-context responses | Suggest related topics from available chunks |

