from langchain_openai import OpenAIEmbeddings
#from langchain_community.retrievers import MultiQueryRetriever
#from langchain_community.retrievers.document_compressors import LLMChainExtractor
#from langchain_community.retrievers import ContextualCompressionRetriever
embeddings = OpenAIEmbeddings()



def get_simple_retriever(vectorstore, request, file_id=None):
    filter_dict = None

    if file_id:
        if isinstance(file_id, list):
            filter_dict = {"file_id": {"$in": file_id}}
        else:
            filter_dict = {"file_id": file_id}

    # 🔥 Dynamic retrieval strategy
    if request.research_mode == 1:
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,   # slightly better than 4
                "fetch_k": 20,  # 🔥 important for MMR quality
                "filter": filter_dict
            }
        )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 2,
            "filter": filter_dict
        }
    )

def get_mmr_retriever(vectorstore, file_id=None):
    filter_dict = None

    if file_id:
        if isinstance(file_id, list):
            filter_dict = {"file_id": {"$in": file_id}}
        else:
            filter_dict = {"file_id": file_id}

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "filter": filter_dict
        }
    )


def get_multiquery_retriever(vectorstore, llm, file_id=None):

    # -----------------------------
    # 1. Build metadata filter
    # -----------------------------
    filter_dict = None

    if file_id:
        if isinstance(file_id, list):
            filter_dict = {"file_id": {"$in": file_id}}
        else:
            filter_dict = {"file_id": file_id}

    # -----------------------------
    # 2. Base MMR retriever
    # -----------------------------
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 15,
            "filter": filter_dict
        }
    )

    # -----------------------------
    # 3. Multi-query generator
    # -----------------------------
    def generate_queries(query: str):
        prompt = f"""
        You are a query expansion system.

        Generate 3 different versions of this query for better document retrieval:

        Query: {query}

        Return only the queries, one per line.
        """

        response = llm.invoke(prompt)
        queries = response.content.strip().split("\n")

        return [q.strip("- ").strip() for q in queries if q.strip()]

    # -----------------------------
    # 4. Final retriever function
    # -----------------------------
    def retrieve(query: str):
        expanded_queries = generate_queries(query)

        all_docs = []

        # include original query too
        expanded_queries.append(query)

        for q in expanded_queries:
            docs = base_retriever.invoke(q)
            all_docs.extend(docs)

        # -----------------------------
        # 5. Deduplicate results
        # -----------------------------
        seen = set()
        unique_docs = []

        for doc in all_docs:
            key = doc.page_content[:200]  # lightweight dedup key
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        return unique_docs

    return retrieve

#-----------------------------------------------------------------------------------------------------
def get_retriever(mode, vectorstore, llm,active_files,request):
    if mode == "qa":
        return get_simple_retriever(vectorstore,request,active_files)

    elif mode == "generation":
        return get_mmr_retriever(vectorstore,active_files)

    elif mode == "research":
        retriever = get_multiquery_retriever(vectorstore, llm,active_files)


        return retriever