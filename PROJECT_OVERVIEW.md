# Project Overview: Unity RAG Bot

This project is a Next.js-based AI Chatbot designed to answer questions about Unity development using RAG (Retrieval-Augmented Generation). It leverages LangChain and the Vercel AI SDK to stream responses and manage conversation state.

## 🛠️ Technology Stack

- **Framework**: [Next.js](https://nextjs.org/) (App Router)
- **Language**: TypeScript
- **AI/LLM**: [LangChain.js](https://js.langchain.com/), [Vercel AI SDK](https://sdk.vercel.ai/)
- **Vector DB**: [Supabase](https://supabase.com/) (pgvector)
- **Models**: OpenAI (GPT-3.5/4), Cohere (Reranking)
- **Styling**: Tailwind CSS, Shadcn/UI

## 📂 Key Directories & Files

### `/app`
Contains the application routes and API endpoints.
- **`/api/chat`**: The main chat endpoint.
    - `route.ts`: Core logic handling chat requests. Implements query transformation, hybrid search (keyword + semantic), reranking, and response generation.
    - Updated to use `LangChainAdapter` for `ai` SDK v3+ compatibility.
- **`/api/retrieval/ingest`**: Endpoint for ingesting documents into the vector store.
- **`/api/*`**: Contains other experimental or legacy endpoints (agents, structured_output) which are currently hidden from navigation.

### `/components`
React components for the UI.
- **`ChatWindow.tsx`**: The main chat interface. Handles message history, input, and displaying intermediate steps (RAG sources).
- **`Navbar.tsx`**: Navigation menu (modified to show only "Chat").
- **`ui/`**: Reusable UI primitives (buttons, dialogs, toasts) based on Shadcn/UI.

### `/scripts`
Node.js scripts for data processing.
- **`ingest-local.mjs`**: Script to ingest local markdown/text files.
- **`ingest-unity.mjs`**: Specialized script for ingesting Unity documentation, likely handling specific HTML structures or CSS classes.

## 🚀 Core Features

1.  **Hybrid RAG Pipeline**:
    - **Query Transformation**: Generates multiple search queries from a single user prompt to improve coverage.
    - **Hybrid Search**: Combines semantic vector search with keyword search (likely using Supabase RPC).
    - **Reranking**: Uses Cohere to re-rank retrieved documents for higher relevance before sending to the LLM.
2.  **Streaming Responses**: Uses Vercel AI SDK to stream LLM responses token-by-token to the client.
3.  **Source Attribution**: Displays the source documents used to generate the answer.

## 🔄 Recent Changes
- **AI SDK Upgrade**: Replaced deprecated `StreamingTextResponse` with `LangChainAdapter`.
- **Navigation Update**: Simplified the main menu to focus solely on the "Chat" experience, hiding experimental features.
