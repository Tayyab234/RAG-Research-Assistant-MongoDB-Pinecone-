from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks,Depends
import os
import uuid
from jwt_hash import hash_password,verify_password,get_current_user,create_token
from mongodb import users,file_data_collection,user_files_collection
from Model import deepseek_llm,llm
from typing_extensions import Annotated
from typing import List
from pydantic_models import UserSignup,UserLogin,UserResponse,QueryRequest
from utility_functions import process_in_background,analyze_and_optimize,retrieval_layer,generate_answer

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE_MB = 20        
#______________________________________________________________________________________________________________________
@app.on_event("startup")
async def startup_db_indexes():
    await file_data_collection.create_index("file_id", unique=True)
    await file_data_collection.create_index("user_id")

@app.post("/signup", response_model=UserResponse)
async def signup(user: UserSignup):
    # Check if user exists
    existing = await users.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_dict = user.dict()
    user_dict["password"] = hash_password(user.password)  # hash password
    await users.insert_one(user_dict)

    return UserResponse(**user_dict)

#______________________________________________________________________________________________________________________
@app.post("/login")
async def login_oauth2(user:UserLogin):
    existing = await users.find_one({"email": user.email})
    if not existing:
        raise HTTPException(status_code=404, detail="User not found")
    if not verify_password(user.password, existing["password"]):
        raise HTTPException(status_code=401, detail="Invalid password")
    
    token = create_token({"user_id": str(existing["_id"])})

    return {
        "access_token": token,
        "token_type": "bearer"
    }

#______________________________________________________________________________________________________________________

@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    files: Annotated[List[UploadFile], File(...)],
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    uploaded_files = []

    for file in files:   # ✅ FIX: loop over files
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, file_id + "_" + file.filename)

        content = await file.read()

        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            continue  # skip large files

        with open(file_path, "wb") as f:
            f.write(content)
            
        # background processing
        background_tasks.add_task(
            process_in_background,
            file_id,
            file_path,
            file.filename,
            user_id
        )

        uploaded_files.append({
            "file_id": file_id,
            "filename": file.filename
        })

    return {
        "message": "Files uploaded. Processing started in background.",
        "files": uploaded_files
    }
@app.get("/status/{file_id}")
async def get_status(file_id: str):
    # ✅ Await the async MongoDB call
    document = await file_data_collection.find_one(
        {"file_id": file_id},
        {"_id": 0, "status": 1, "num_chunks": 1, "total_tokens": 1}  # only fetch metadata
    )

    if not document:
        raise HTTPException(status_code=404, detail="File not found")

    return {"file_id": file_id, **document}

@app.get("/files")
async def get_user_files(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]

    files = await user_files_collection.find(
        {"user_id": user_id},
        {"_id": 0}
    ).to_list(length=100)

    return {
        "files": files
    }

@app.post("/set-active-files")
async def set_active_files(
    file_ids: List[str],
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]

    # ❌ First deactivate all
    await user_files_collection.update_many(
        {"user_id": user_id},
        {"$set": {"is_active": False}}
    )

    # ✅ Activate selected files
    await user_files_collection.update_many(
        {
            "user_id": user_id,
            "file_id": {"$in": file_ids}
        },
        {"$set": {"is_active": True}}
    )

    return {"message": "Active files updated"}

@app.post("/add-active-files")
async def add_active_files(
    file_ids: List[str],
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]

    await user_files_collection.update_many(
        {
            "user_id": user_id,
            "file_id": {"$in": file_ids}
        },
        {"$set": {"is_active": True}}
    )

    return {"message": "Files added to active list"}

#------------------------------------------------------------------------------------------------------------------
@app.post("/query")
async def query_endpoint(request: QueryRequest,current_user: dict = Depends(get_current_user)):
     user_id = current_user["user_id"]
     query_optimization = request.query_optimization

     if request.research_mode == 1:
        query_optimization = 1
        
     #layer_1
     analysis=  analyze_and_optimize(request.query, llm,request.query_optimization)
     context_data=await retrieval_layer(analysis,llm,user_id,request)
     if analysis["scope"]=="full":
         return context_data
         
     return await generate_answer(analysis, context_data, llm, request)
