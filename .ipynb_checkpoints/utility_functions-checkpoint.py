from document_processor import process_document,estimate_tokens
from pineconeDB import upsert_vectors,vectorstore_setup
from collections import defaultdict
from mongodb import file_data_collection,user_files_collection
import asyncio
import traceback
from Model import embeddings
import os
from prompts import query_analyzer,llm_prompt,QA_PROMPT_STRICT,QA_PROMPT_FLEX,RESEARCH_PROMPT,GENERATION_PROMPT
from pydantic_models import QueryAnalysis
from Map_reduce import map_reduce_pipeline
from retrievers import get_retriever 
import datetime
from datetime import timedelta,datetime

async def insert_initial_doc(user_id, file_id, filename):
    initial_doc = {
        "user_id": user_id,
        "file_id": file_id,
        "filename": filename,
        "status": "processing",
        "chunks": []
    }
    await file_data_collection.insert_one(initial_doc)


async def store_chunks_and_finish(file_id, chunks, total_tokens):
    await file_data_collection.update_one(
        {"file_id": file_id},
        {"$set": {
             "chunks": [c.page_content for c in chunks],
             "num_chunks": len(chunks),
             "total_tokens": total_tokens,
             "status": "finished"
        }}
    )



async def upsert_vectors_pinecone(user_id, file_id, chunks):
    """
    user_id: string
    file_id: string
    chunks: list of text chunks
    """

    namespace = user_id
    texts = [chunk.page_content for chunk in chunks]
    # Generate embeddings (run in thread if blocking)
    chunk_embeddings = await asyncio.to_thread(
        embeddings.embed_documents, texts
    )

    vectors = []
    for i, embedding in enumerate(chunk_embeddings):
        vector = {
            "id": f"{file_id}_{i}",  # unique ID per chunk
            "values": embedding,
            "metadata": {
                "file_id": file_id,
                "chunk_id": i
            }
        }
        vectors.append(vector)

    return await upsert_vectors(namespace, vectors)

  


async def process_in_background(file_id, file_path, filename, user_id):
    try:
        await insert_initial_doc(user_id, file_id, filename)

        # ---------------------------
        # DEBUG LOGS (VERY IMPORTANT)
        # ---------------------------
        print("🚀 Starting processing:", file_path)
        print("Exists:", os.path.exists(file_path))
        print("Size:", os.path.getsize(file_path))

        # ---------------------------
        # Run sync function safely
        # ---------------------------
        result = await asyncio.to_thread(
            process_document,
            file_path,
            filename
        )

        print("✅ Document processed successfully")

        await store_chunks_and_finish(
            file_id,
            result["chunks"],
            result["total_tokens"]
        )

        await upsert_vectors_pinecone(
            user_id,
            file_id,
            result["chunks"]
        )

        # ✅ Store in MongoDB
        await user_files_collection.insert_one({
            "user_id": user_id,
            "file_id": file_id,
            "filename": filename,
            "is_active": True,  # default active
            "created_at": datetime.utcnow()
        })

        if os.path.exists(file_path):
            os.remove(file_path)

    except Exception as e:
        print("❌ ERROR FULL TRACEBACK:")
        print(traceback.format_exc())

        await file_data_collection.update_one(
            {"file_id": file_id},
            {
                "$set": {
                    "status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            }
        )
async def get_active_file_ids(user_id: str):
    files = await user_files_collection.find(
        {
            "user_id": user_id,
            "is_active": True
        },
        {
            "_id": 0,
            "file_id": 1
        }
    ).to_list(length=100)

    return [f["file_id"] for f in files]

def analyze_and_optimize(query, llm, optimize_flag):

    prompt=query_analyzer.invoke({"query":query,"optimize_flag":optimize_flag})
    model=llm.with_structured_output(QueryAnalysis)
    response = model.invoke(prompt)
    return   {
    "Query":response.Query,
    "scope":response.scope,
    "mode":response.mode
    }
#----------------------------------------------------------------------------------------------------------------------
FULL_DOC_LIMIT = 50000   # safe for most LLMs

MAX_ALLOWED_DOC_TOKENS = 1500000

async def get_user_documents(user_id: str, active_files):
    if not active_files:
        return []

    files = await user_files_collection.find(
        {
            "user_id": user_id,
            "file_id": {"$in": active_files}
        },
        {"_id": 0}
    ).to_list(length=100)

    return files
    
async def process_full_documents(analysis, user_id, llm,request):
    active_files = await get_active_file_ids(user_id)
    file_docs = await get_user_documents(user_id,active_files)

    results = []

    for file_doc in file_docs:
        chunks = file_doc.get("chunks", [])
        file_name = file_doc.get("filename", "unknown")

        full_text = " ".join(chunks)

        total_tokens = estimate_tokens(full_text)

        if total_tokens > MAX_ALLOWED_DOC_TOKENS:
            return {
                "error": f"Document '{file_name}' is too large. Please specify a topic within the document."
            }

        elif total_tokens > FULL_DOC_LIMIT:
            # 🔴 Too large → Map Reduce
            result = await map_reduce_pipeline(
                chunks, analysis, llm, file_name
            )
            results.append({
                "file_name": file_name,
                "result": result
            })
            
        else:
            context_data= {
                "context": full_text,
                "sources": file_name
            }
            
            result= await generate_answer(analysis,context_data, llm, request)

            results.append({
                "file_name": result["sources"],
                "result": result["response"]
            })

    return results

#----------------------------------------------------------------------------------------------------------------------

async def retrieval_layer(analysis, llm, user_id,request):
    vectorstore = vectorstore_setup(user_id)

    active_files = await get_active_file_ids(user_id)

    if analysis["scope"] == "full":
        return await process_full_documents(analysis, user_id, llm,request)
    
    retriever = get_retriever(
        mode=analysis["mode"],
        vectorstore=vectorstore,
        llm=llm,
        active_files=active_files,
        request=request
    )
    
   # Step 1: retrieve from Pinecone
    docs = retriever.invoke(analysis["Query"])
    if not docs:
        return {"context": [], "sources": []}
    
    # Step 2: extract references
    refs = [
        (doc.page_content, doc.metadata.get("chunk_id"))
        for doc in docs
    ]
    
    # Step 3: group by file_id
    grouped = defaultdict(set)
    
    for file_id, chunk_id in refs:
        if chunk_id is not None:
            grouped[file_id].add(chunk_id)
    
    # Step 4: fetch from MongoDB
    final_context = []
    sources = set()
    
    for file_id, chunk_ids in grouped.items():
        file_doc = await file_data_collection.find_one(
            {"file_id": file_id},
            {"chunks": 1, "filename": 1, "_id": 0}
        )
    
        if not file_doc or "chunks" not in file_doc:
            continue
    
        chunks = file_doc["chunks"]
        file_name = file_doc.get("filename", file_id)  # fallback
    
        for cid in chunk_ids:
            try:
                cid_int = int(cid)
            except (ValueError, TypeError):
                continue
    
            if 0 <= cid_int < len(chunks):
                chunk = chunks[cid_int]
                if isinstance(chunk, str):
                    final_context.append(chunk)
                    sources.add(file_name)  # ✅ add readable source
    
    return {
        "context": final_context,
        "sources": list(sources)
    }
    
def get_prompt(mode, strict_mode):
    if mode == "qa":
        return QA_PROMPT_STRICT if strict_mode else QA_PROMPT_FLEX

    elif mode == "generation":
        return GENERATION_PROMPT

    elif mode == "research":
        return RESEARCH_PROMPT

    else:
        return QA_PROMPT_FLEX
    
async def generate_answer(analysis, context_data, llm, request):
    context = "\n\n".join(context_data["context"])
    query=analysis["Query"]
    if request.research_mode == 1:
        analysis.mode = "research"
    strict_mode_text = "ENABLED" if request.strict_mode == 1 else "DISABLED" 

    system_prompt = get_prompt(analysis["mode"], request.strict_mode)
    formatted_prompt = llm_prompt.format(system_prompt=system_prompt,strict_mode=strict_mode_text,
                                         context=context,query=analysis["Query"])

    response = await llm.ainvoke(formatted_prompt)


    return {
        "response": response.content,
        "sources": context_data["sources"]
    }




