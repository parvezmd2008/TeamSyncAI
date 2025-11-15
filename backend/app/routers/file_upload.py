# app/routers/file_upload.py

from fastapi import APIRouter, UploadFile, File, HTTPException, status
import shutil
import os
import aiofiles

router = APIRouter()

# Directory to temporarily store uploaded files before LangChain processing
UPLOAD_DIR = "uploaded_data" 
os.makedirs(UPLOAD_DIR, exist_ok=True) # Ensure the directory exists

@router.post("/upload")
async def upload_document(
    files: list[UploadFile] = File(
        ..., 
        description="List of files to upload (WhatsApp chat, Google Meet transcript, etc.)"
    )
):
    """
    Handles the upload of documents for AI processing.
    The files are streamed to disk before being handed off for LangChain processing.
    """
    
    uploaded_filenames = []
    
    for file in files:
        # Validate file type if necessary (e.g., .txt, .json, .pdf)
        # You can add logic here to check file.content_type
        
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        
        try:
            # Asynchronously stream the uploaded file to disk
            async with aiofiles.open(file_location, "wb") as out_file:
                # Read chunks of the file to save memory
                while content := await file.read(1024 * 1024): # Read in 1MB chunks
                    await out_file.write(content)
            
            uploaded_filenames.append(file.filename)

            # --- Placeholder for LangChain Integration (Your Colleague's Part) ---
            # NOTE: After saving, you would typically call a service function 
            #       to start the LangChain process, passing 'file_location'.
            
            # Example: 
            # from app.services import llm_service
            # await llm_service.process_file_for_rag(file_location, file.content_type)
            
            # The LangChain process would likely:
            # 1. Load the document.
            # 2. Split it into chunks.
            # 3. Create embeddings.
            # 4. Store the embeddings in a Vector Database (e.g., MongoDB/Vector Search).
            # 5. Kick off a task to generate To-Do list/Important Dates.
            # -------------------------------------------------------------------
            
        except Exception as e:
            # It's good practice to close the file if an error occurs
            await file.close()
            # Log the error and return an appropriate status
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An error occurred during the upload of {file.filename}: {e}"
            )
        finally:
            # The file is typically closed automatically, but good to be explicit 
            # if your colleague is handling file content directly
            pass

    return {
        "status": "success",
        "message": "Files uploaded successfully for processing.",
        "filenames": uploaded_filenames
    }