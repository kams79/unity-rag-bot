import { NextRequest, NextResponse } from "next/server";
import { LangChainAdapter } from "ai";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { createClient } from "@supabase/supabase-js";
// --- FIXED IMPORTS START ---
import { StringOutputParser } from "@langchain/core/output_parsers";
// import { HttpResponseOutputParser } from "langchain/output_parsers";
// --- FIXED IMPORTS END ---
import { RunnableSequence } from "@langchain/core/runnables";
import { CohereRerank } from "@langchain/cohere";
import { Document } from "@langchain/core/documents";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const messages = body.messages ?? [];
    const currentMessageContent = messages[messages.length - 1].content;

    if (!process.env.OPENROUTER_API_KEY) throw new Error("Missing OpenRouter Key");
    if (!process.env.COHERE_API_KEY) throw new Error("Missing Cohere Key");

    const client = createClient(
      process.env.SUPABASE_URL!,
      process.env.SUPABASE_PRIVATE_KEY!,
      { auth: { persistSession: false } }
    );

    const embeddings = new OpenAIEmbeddings({
      modelName: "openai/text-embedding-3-small",
      apiKey: process.env.OPENROUTER_API_KEY,
      configuration: { baseURL: "https://openrouter.ai/api/v1" },
    });

    const model = new ChatOpenAI({
      modelName: "mistralai/mistral-7b-instruct",
      temperature: 0,
      apiKey: process.env.OPENROUTER_API_KEY,
      configuration: { baseURL: "https://openrouter.ai/api/v1" },
    });

    const cohereRerank = new CohereRerank({
      apiKey: process.env.COHERE_API_KEY,
      model: "rerank-english-v3.0",
    });

    // --- PHASE 1: QUERY TRANSFORMATION ---
    const queryGenerationPrompt = PromptTemplate.fromTemplate(`
      You are a specialized Unity AI assistant. 
      Your task is to generate 3 different search queries for the Unity Documentation based on the user's question.
      
      RULES:
      1. One query should be the exact user question.
      2. One query should focus on specific Unity classes or keywords.
      3. One query should be a "how-to" phrasing.
      4. Output ONLY the 3 queries separated by newlines. No numbering.
      
      User Question: {question}
      
      Queries:
    `);

    const queryChain = RunnableSequence.from([
      queryGenerationPrompt,
      model,
      new StringOutputParser(), // Uses @langchain/core
    ]);

    console.log("🤔 Generating query variations...");
    const rawQueries = await queryChain.invoke({ question: currentMessageContent });
    const queries = rawQueries.split('\n').filter(q => q.trim() !== "");
    console.log("💡 Generated Queries:", queries);

    // --- PHASE 2: MULTI-QUERY HYBRID SEARCH ---
    const multiQueryRetriever = async () => {
      const searchPromises = queries.map(async (q) => {
        const vector = await embeddings.embedQuery(q);
        const { data } = await client.rpc("hybrid_search", {
          query_text: q,
          query_embedding: vector,
          match_count: 10,
          full_text_weight: 1.0,
          semantic_weight: 1.0,
          rrf_k: 60,
        });
        return data || [];
      });

      const results = await Promise.all(searchPromises);
      const allDocs = results.flat();

      const uniqueDocsMap = new Map();
      allDocs.forEach((doc: any) => {
        if (!uniqueDocsMap.has(doc.id)) {
          uniqueDocsMap.set(doc.id, new Document({
            pageContent: doc.content,
            metadata: doc.metadata
          }));
        }
      });

      return Array.from(uniqueDocsMap.values());
    };

    // --- PHASE 3: RERANKING ---
    const advancedRetriever = async () => {
      const uniqueDocs = await multiQueryRetriever();
      console.log(`📚 Found ${uniqueDocs.length} unique docs from multi-query.`);

      if (uniqueDocs.length === 0) return "";

      const reranked = await cohereRerank.rerank(uniqueDocs, currentMessageContent, {
        topN: 5
      });

      return reranked.map(r => uniqueDocs[r.index].pageContent).join("\n\n---\n\n");
    };

    // --- PHASE 4: GENERATION ---
    const answerPrompt = PromptTemplate.fromTemplate(`
      You are a specialized Unity Documentation Assistant.
      
      STRICT RULES:
      1. Use ONLY the provided context.
      2. If the answer is not in the context, say "I don't know".
      3. If the context contains code snippets (C#), prioritize showing them.
      
      Context:
      {context}
      
      Question: 
      {question}
      
      Answer:
    `);

    const chain = RunnableSequence.from([
      {
        context: async () => await advancedRetriever(),
        question: (input: string) => input,
      },
      answerPrompt,
      model,
      new StringOutputParser(),
    ]);

    const stream = await chain.stream(currentMessageContent);
    return LangChainAdapter.toDataStreamResponse(stream);

  } catch (e: any) {
    console.error(e);
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}