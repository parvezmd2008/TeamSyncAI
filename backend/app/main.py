# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import file_upload, llm_agent
import os

# You should define this list based on where your React app is running.
# In development, it's often http://localhost:3000 or http://localhost:5173
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    # Add your production domain when you deploy
]

app = FastAPI(
    title="TeamSync AI Productivity API",
    version="0.1.0",
    description="Backend API for TeamSync using FARM Stack and Gemini.",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,              # Allows specific origins
    allow_credentials=True,             # Allows cookies/auth headers
    allow_methods=["*"],                # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],                # Allows all headers
)

# --- Include Routers ---
app.include_router(file_upload.router, prefix="/api/v1/files", tags=["File Upload"])
app.include_router(llm_agent.router, prefix="/api/v1/agent", tags=["AI Agent & Chat"])

# --- Health Check Endpoint (Optional but Recommended) ---
@app.get("/")
async def root():
    return {"message": "TeamSync API is running!"}

# You will run the app using Uvicorn: 
# uvicorn app.main:app --reload --port 8000