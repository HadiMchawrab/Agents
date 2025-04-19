from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import traceback
import logging
from virtualgraph import run_graph
from virtualgraph_v2 import run_graph_v2
from typing import List
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

UPLOAD_FOLDER = 'csv_test'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class InitialState(BaseModel):
    csv_files: List[str]

@app.get("/")
async def index():
    return "Hello, World!"

@app.post("/upload-and-process")
async def upload_and_process(files: List[UploadFile] = File(...)):
    try:
        logger.debug("Received upload request")
        
        # Save uploaded files
        logger.debug(f"Received {len(files)} files")
        
        csv_files = []
        
        for file in files:
            if file.filename.endswith('.csv'):
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                logger.debug(f"Saving file to {file_path}")
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                csv_files.append(file_path)

        # Execute the graph with the initial state
        logger.debug("Starting graph execution")
        result = run_graph(csv_files)
        logger.debug("Graph execution completed")

        return {
            'status': 'success',
            'message': 'Files processed successfully',
            'result': result
        }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                'status': 'error',
                'message': str(e),
                'traceback': traceback.format_exc()
            }
        )


@app.post("/upload-and-process-v2")
async def upload_and_process_v2(files: List[UploadFile] = File(...)):
    try:
        logger.debug("Received upload request")
        
        # Save uploaded files
        logger.debug(f"Received {len(files)} files")
        
        csv_files = []
        
        for file in files:
            if file.filename.endswith('.csv'):
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                logger.debug(f"Saving file to {file_path}")
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                csv_files.append(file_path)

        # Execute the graph with the initial state
        logger.debug("Starting graph execution")
        result = run_graph_v2(csv_files)
        logger.debug("Graph execution completed")

        return {
            'status': 'success',
            'message': 'Files processed successfully',
            'result': result
        }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                'status': 'error',
                'message': str(e),
                'traceback': traceback.format_exc()
            }
        )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)