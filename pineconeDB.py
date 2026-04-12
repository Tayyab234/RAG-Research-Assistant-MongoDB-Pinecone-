import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from Model import embeddings
import random
import asyncio

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX", "main-index")

# ✅ Create a Pinecone client instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # adjust to your embedding model
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_ENV
        )
    )

desc = pc.describe_index(INDEX_NAME)
index = pc.Index(host=desc.host)



# Reusable upsert function
async def upsert_vectors(namespace: str, vectors: list):
    try:
        await asyncio.to_thread(
            index.upsert,
            vectors=vectors,
            namespace=namespace
        )

        return {
            "status": "success",
            "num_vectors": len(vectors)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


#__________________________________________________________________________________________________________


def vectorstore_setup(user_id:str):
    # Create vector store
     return  PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=user_id,
        text_key="file_id"
    )