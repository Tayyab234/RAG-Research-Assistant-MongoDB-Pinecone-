from pydantic import BaseModel, EmailStr,Field
from typing import Optional,List

# For signup
class UserSignup(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    age: Optional[int] = Field(None, ge=10, le=100)

# For login
class UserLogin(BaseModel):
    email: EmailStr
    password: str
class UserResponse(BaseModel):
    username: str
    email: EmailStr
    age: Optional[int]

class QueryAnalysis(BaseModel):
    Query: str=Field(..., description="Query of the user either optimized version or original version")
    scope: str  # full | partial
    mode: str   # qa | generation | research
    
class QueryRequest(BaseModel):
    query: str
    query_optimization: Optional[int] = 0
    strict_mode: Optional[int] = 0
    research_mode: Optional[int] = 0