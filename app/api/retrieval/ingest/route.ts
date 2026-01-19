import { NextRequest, NextResponse } from "next/server";
import { Message as VercelChatMessage, LangChainAdapter } from "ai";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { createClient } from "@supabase/supabase-js";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
import { formatDocumentsAsString } from "langchain/util/document";

export const runtime = "edge";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const messages = body.messages ?? [];
    const currentMessageContent = messages[messages.length - 1].content;

    // 1. Init Vector Store
    const client = createClient(
      process.env.SUPABASE_URL!,
      process.env.SUPABASE_PRIVATE_KEY!
    );
    const vectorStore = new SupabaseVectorStore(new OpenAIEmbeddings(), {
      client,
      tableName: "documents",
      queryName: "match_documents",
    });

    // 2. Retriever
    const retriever = vectorStore.asRetriever(4);

    // 3. LLM (Temperature 0 for strictness)
    const model = new ChatOpenAI({
      modelName: "gpt-3.5-turbo",
      temperature: 0,
    });

    // 4. Strict Prompt
    const prompt = PromptTemplate.fromTemplate(`
      You are a specialized Unity Documentation Assistant.
      
      STRICT RULES:
      1. Use ONLY the provided context to answer the question.
      2. If the answer is not in the context, you MUST say "I don't know".
      3. Do not make up code or classes that are not in the context.
      
      Context:
      {context}
      
      Question: 
      {question}
      
      Answer:
    `);

    // 5. Chain
    const chain = RunnableSequence.from([
      {
        context: async (input: string) => {
          const relevantDocs = await retriever.getRelevantDocuments(input);
          console.log(`Found ${relevantDocs.length} docs`);
          return formatDocumentsAsString(relevantDocs);
        },
        question: (input: string) => input,
      },
      prompt,
      model,
      new StringOutputParser(),
    ]);

    const stream = await chain.stream(currentMessageContent);
    return LangChainAdapter.toDataStreamResponse(stream);

  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}