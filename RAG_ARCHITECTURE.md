# RAG Architecture: Advanced Multi-Stage Pipeline

The Retrieval-Augmented Generation (RAG) system implemented in `app/api/chat/route.ts` is designed for high accuracy and robustness. Unlike naive RAG approaches, it employs a multi-stage pipeline to ensure that the context provided to the LLM is both relevant and comprehensive.

## Pipeline Phases

### 1. Query Transformation (Multi-Query)
**Goal:** overcoming the limitations of a single, potentially poorly phrased user query.

-   **Mechanism:** An LLM (using `PromptTemplate` and `ChatOpenAI`) analyzes the user's raw question and generates **3 distinct search queries**:
    1.  **Exact**: The original user question.
    2.  **Keywords**: Focused on specific Unity classes or technical terms.
    3.  **How-to**: Phrases the question as a "how-to" intent.
-   **Benefit:** Increases the likelihood of retrieving relevant documents even if the user's initial wording doesn't match the terminology in the documentation.

### 2. Hybrid Search (Keyword + Semantic)
**Goal:** retrieving a broad set of potentially relevant candidates.

-   **Mechanism:** All 3 queries are executed in parallel against the Supabase vector store using a `hybrid_search` RPC function.
    -   **Semantic Search:** Uses OpenAI embeddings (`text-embedding-3-small`) to find documents that conceptually match the queries.
    -   **Full-Text Search:** Uses keyword matching (likely BM25-style) to find exact term matches.
    -   **Reciprocal Rank Fusion (RRF):** The results from both methods are combined and ranked (indicated by `rrf_k: 60`), ensuring a balanced mix of semantically similar and keyword-exact documents.

### 3. Reranking (Cohere)
**Goal:** filtering the retrieved candidates to find the "true" best matches.

-   **Mechanism:** The deduplicated results from the hybrid search are passed to **Cohere Rerank** (`rerank-english-v3.0`).
-   **Process:** Cohere's model evaluates the relevance of each document-query pair and assigns a relevance score.
-   **Output:** Only the **top 5** highest-scoring documents are selected for the final generation phase. This step drastically reduces "noise" and distracting context.

### 4. Generation (Context-Aware)
**Goal:** synthesizing the answer.

-   **Mechanism:** The top 5 reranked documents are formatted into a prompt for the final LLM (`mistral-7b-instruct` or similar).
-   **Instructions:** The system strictly instructs the model to:
    -   Use *only* the provided context.
    -   Prioritize code examples (C#).
    -   Admit ignorance if the answer isn't in the context.
-   **Streaming:** The response is streamed back to the client token-by-token using the `LangChainAdapter` (formerly `StreamingTextResponse`) for a responsive user experience.
