
import dotenv from 'dotenv';
import path from 'path';
import fs from 'fs/promises';
import { createClient } from "@supabase/supabase-js";
import { OpenAIEmbeddings } from "@langchain/openai";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as cheerio from 'cheerio';
import { glob } from 'glob';

// Load environment variables from .env.local
dotenv.config({ path: '.env.local' });

// Configuration
const DOCS_ROOT = path.resolve('../Documentation/en'); // Adjust path as needed
const BASE_URL = "https://docs.unity3d.com/6000.0/Documentation";
const BATCH_SIZE = 50;

async function run() {
    // CLI Arguments
    const args = process.argv.slice(2);
    const isDryRun = args.includes('--dry-run');
    const limitArg = args.find(a => a.startsWith('--limit='));
    const limit = limitArg ? parseInt(limitArg.split('=')[1]) : Infinity;

    console.log("🚀 Starting Local Unity Docs Ingestion...");
    console.log(`📂 Reading from: ${DOCS_ROOT}`);
    if (isDryRun) console.log("🧪 DRY RUN MODE: DB will not be updated.");
    if (limit !== Infinity) console.log(`🔢 Limit: Processing max ${limit} files.`);

    // --- SETUP SUPABASE & EMBEDDINGS ---
    let client, embeddings;
    if (!isDryRun) {
        if (!process.env.OPENROUTER_API_KEY) throw new Error("Missing OpenRouter Key");
        if (!process.env.SUPABASE_URL || !process.env.SUPABASE_PRIVATE_KEY) throw new Error("Missing Supabase Credentials");

        client = createClient(
            process.env.SUPABASE_URL,
            process.env.SUPABASE_PRIVATE_KEY,
            { auth: { persistSession: false } }
        );

        embeddings = new OpenAIEmbeddings({
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
    }

    // --- FILE DISCOVERY ---
    // We want Manual and ScriptReference
    const pattern = `${DOCS_ROOT}/@(Manual|ScriptReference)/**/*.html`;
    let files = await glob(pattern);


    console.log(`🔎 Found ${files.length} HTML files.`);

    if (limit < Infinity) {
        files = files.slice(0, limit);
    }

    let documents = [];

    // --- PROCESSING LOOP ---
    for (const filePath of files) {
        try {
            const content = await fs.readFile(filePath, 'utf-8');
            const $ = cheerio.load(content);
            const relativePath = path.relative(DOCS_ROOT, filePath);
            const docType = relativePath.startsWith('Manual') ? 'Manual' : 'ScriptReference';

            // Construct Canonical URL
            const url = `${BASE_URL}/${relativePath}`;

            // 1. Title Extraction
            let title = $('h1').first().text().trim();
            if (!title) title = $('title').text().replace(' - Unity Manual', '').replace(' - Unity Scripting API', '').trim();

            // 2. Description/Summary Extraction
            let description = "";
            if (docType === 'Manual') {
                // Manual usually has the first paragraph in .content or standard p tags
                description = $('.content p').first().text().trim();
            } else {
                // ScriptReference often has a description block
                description = $('.subsection').has('h3:contains("Description")').find('p').text().trim();
                // Fallback
                if (!description) description = $('.subsection p').first().text().trim();
            }

            // 3. Code Snippets Extraction
            // Strategy: Extract code blocks into separate Documents to preserve them as "whole code".
            // Then remove them from the DOM so they don't get chopped up in the main text.

            const codeDocs = [];

            // Helper to process code elements
            const processCodeBlock = (index, element, typeLabel) => {
                const $el = $(element);
                const codeText = $el.text().trim();
                if (codeText.length > 10) { // Ignore tiny snippets
                    codeDocs.push({
                        pageContent: codeText,
                        metadata: {
                            source: url,
                            title: `${title} (Code Snippet ${index + 1})`,
                            type: 'code_snippet', // Distinct type
                            original_type: docType,
                            description: `Code example from ${title}`
                        }
                    });
                }
                // Remove from DOM to avoid duplication in main text
                // For pre > code, we want to remove the pre parent usually
                if ($el.parent().is('pre')) {
                    $el.parent().remove();
                } else {
                    $el.remove();
                }
            };

            // Selector A: ScriptReference style
            $('.codeExampleCS').each((i, el) => processCodeBlock(i, el, 'ScriptReference'));

            // Selector B: Manual style (pre > code.lang-cs, code.lang-csharp)
            // Note: We use a broad selector for 'pre > code' and verify class or just take it if it looks like code
            $('pre > code').each((i, el) => {
                processCodeBlock(i + 100, el, 'Manual'); // offset index to avoid collision if mixed
            });

            // Add code docs to main list
            documents.push(...codeDocs);

            // 4. Main Body Content Cleaning
            // Remove navigation, header, footer, scripts
            $('script').remove();
            $('style').remove();
            $('.header-wrapper').remove();
            $('.footer-wrapper').remove();
            $('.sidebar').remove();
            $('.toc').remove();
            $('.mb20.clear').remove(); // Breadcrumbs/Next-Prev nav
            $('.subsection h3').remove(); // "Description" headers etc usually add noise if repetitive

            let cleanText = $('.content-block').text().replace(/\s\s+/g, ' ').trim();

            // For ScriptReference, sometimes the signature is important
            // It's usually in div.signature
            const signature = $('.signature-CS').text().trim();
            if (signature) {
                cleanText = `Signature:\n${signature}\n\n${cleanText}`;
            }



            // Create Document Object (compatible with LangChain)
            documents.push({
                pageContent: cleanText,
                metadata: {
                    source: url,
                    title: title,
                    type: docType,
                    description: description.substring(0, 300), // Limit meta description length
                    has_code: codeDocs.length > 0
                }
            });

            // 5. Special Handling for Code Snippets
            // If there are large code snippets, we might want to index them as separate chunks associated with the same page
            // For now, we are relying on the text splitter to keep them somewhat intact, but let's append them explicitly to text if they were stripped or hard to parse,
            // or just rely on the fact that cheerio .text() includes them.
            // Cheerio .text() DOES include them, but we might want to format them better. 
            // The `cleanText` above includes code.

        } catch (e) {
            console.error(`Error processing ${filePath}:`, e);
        }
    }

    console.log(`🧩 Extracted content from ${documents.length} files.`);

    // --- CHUNKING ---
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 2000,
        chunkOverlap: 200,
    });

    const splits = await splitter.splitDocuments(documents);
    console.log(`🔪 Generated ${splits.length} chunks.`);

    // --- UPLOAD ---
    if (!isDryRun) {
        console.log("💾 Uploading to Supabase...");
        for (let i = 0; i < splits.length; i += BATCH_SIZE) {
            const batch = splits.slice(i, i + BATCH_SIZE);
            try {
                await SupabaseVectorStore.fromDocuments(batch, embeddings, {
                    client,
                    tableName: "documents",
                    queryName: "match_documents",
                });
                process.stdout.write(`.`); // Progress dot
            } catch (e) {
                console.error(`\nError uploading batch ${i}:`, e);
            }
        }
        console.log("\n🎉 Upload Complete!");
    } else {
        console.log("🧪 Dry run complete. Sample Chunk:");
        if (splits.length > 0) {
            console.log(splits[0]);
        }
    }
}

run().catch(console.error);
