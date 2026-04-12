import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "rag_app")

# Create client
client = AsyncIOMotorClient(MONGODB_URI)

# Database
db = client[DB_NAME]

# Collections
users = db["users"]
file_data_collection = db["file_data"]
user_files_collection=db["user_files"]