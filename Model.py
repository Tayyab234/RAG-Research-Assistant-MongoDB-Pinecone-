from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os

load_dotenv()

embeddings = OpenAIEmbeddings()

llm = ChatOpenAI(
    model="gpt-4o-mini"
)


# DeepSeek uses OpenAI-compatible API
deepseek_llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com/v1",
)

