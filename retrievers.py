from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

#--------------------------------------------------------------------------------------------------------------------
class MultiQueryRetriever:
    def __init__(self, base_retriever, llm):
        self.base_retriever = base_retriever
        self.llm = llm

    def generate_queries(self, query: str):
        prompt = f"""
        You are a query expansion system.

        Generate 3 different versions of this query:

        Query: {query}

        Return one query per line.
        """
        response = self.llm.invoke(prompt)
        return [q.strip("- ").strip() for q in response.content.split("\n") if q.strip()]

    def invoke(self, query: str):
        expanded_queries = self.generate_queries(query)
        expanded_queries.append(query)

        all_docs = []

        for q in expanded_queries:
            docs = self.base_retriever.invoke(q)
            all_docs.extend(docs)

        # deduplicate
        seen = set()
        unique_docs = []
        
        for doc in all_docs:
            chunk_id = doc.metadata.get("chunk_id")
            content = doc.page_content
        
            key = (chunk_id, content)  # 👈 combine both
        
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        
        return unique_docs

#--------------------------------------------------------------------------------------------------------------------
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

    filter_dict = None

    if file_id:
        if isinstance(file_id, list):
            filter_dict = {"file_id": {"$in": file_id}}
        else:
            filter_dict = {"file_id": file_id}

    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 15,
            "filter": filter_dict
        }
    )

    return MultiQueryRetriever(base_retriever, llm)
#-----------------------------------------------------------------------------------------------------
def get_retriever(mode, vectorstore, llm,active_files,request):
    if mode == "qa":
        return get_simple_retriever(vectorstore,request,active_files)

    elif mode == "generation":
        return get_mmr_retriever(vectorstore,active_files)

    elif mode == "research":
        retriever = get_multiquery_retriever(vectorstore, llm,active_files)


        return retriever