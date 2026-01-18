import dotenv from 'dotenv';
dotenv.config({ path: '.env.local' });

import { createClient } from "@supabase/supabase-js";
import { OpenAIEmbeddings } from "@langchain/openai";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { RecursiveUrlLoader } from "@langchain/community/document_loaders/web/recursive_url";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { compile } from "html-to-text";

async function run() {
    console.log("🚀 Starting Unified Unity Docs Ingestion...");

    if (!process.env.OPENROUTER_API_KEY) throw new Error("Missing OpenRouter Key");

    const client = createClient(
        process.env.SUPABASE_URL,
        process.env.SUPABASE_PRIVATE_KEY,
        { auth: { persistSession: false } }
    );

    // --- EMBEDDINGS (Must match your Chat Route) ---
    const embeddings = new OpenAIEmbeddings({
        modelName: "openai/text-embedding-3-small",
        apiKey: process.env.OPENROUTER_API_KEY,
        configuration: {
            baseURL: "https://openrouter.ai/api/v1",
            defaultHeaders: {
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Unity RAG Bot Ingestion",
            },
        },
    });

    // --- HTML CONVERTER (Tuned for Code Snippets) ---
    // We disable wordwrap for code blocks to keep them clean
    const compiledConvert = compile({
        wordwrap: 130,
        selectors: [
            { selector: 'div.code_snippet', format: 'pre' }, // Force Unity snippets to formatting
            { selector: 'pre', options: { leadingLineBreaks: 1, trailingLineBreaks: 1 } },
            { selector: 'a', options: { ignoreHref: true } }, // Reduce noise
            { selector: 'div.footer-wrapper', format: 'skip' } // Skip footer junk
        ]
    });

    // --- SOURCES TO SCRAPE ---
    const targets = [
        "https://docs.unity3d.com/Manual/ScriptingSection.html", // The Manual (Concepts)
        "https://docs.unity3d.com/ScriptReference/index.html"    // The API (Code Snippets)
    ];

    let allDocs = [];

    for (const url of targets) {
        console.log(`🕷️  Scraping: ${url}`);
        const loader = new RecursiveUrlLoader(url, {
            maxDepth: 10, // Keep shallow for demo speed. Increase to 3+ for full API coverage.
            extractor: compiledConvert,
            // Optional: exclude navigation pages to focus on content
            excludeDirs: ["https://docs.unity3d.com/Manual/BestPracticeGuides"],
            preventOutside: true,
            timeout: 10000,
        });

        const docs = await loader.load();
        console.log(`   - Found ${docs.length} pages.`);
        allDocs = allDocs.concat(docs);
    }

    // --- CHUNKING ---
    // We use a larger chunk size to ensure code snippets stay together
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1500, // Increased size for code blocks
        chunkOverlap: 200,
    });

    const splits = await splitter.splitDocuments(allDocs);
    console.log(`🧩 Total Chunks to Embed: ${splits.length}`);

    // --- UPLOAD ---
    console.log("💾 Uploading to Supabase (this may take a while)...");

    // Batching to avoid timeouts
    const batchSize = 100;
    for (let i = 0; i < splits.length; i += batchSize) {
        const batch = splits.slice(i, i + batchSize);
        await SupabaseVectorStore.fromDocuments(batch, embeddings, {
            client,
            tableName: "documents",
            queryName: "match_documents",
        });
        console.log(`   - Uploaded batch ${i / batchSize + 1}`);
    }

    console.log("🎉 Ingestion Complete!");
}

run().catch(console.error);