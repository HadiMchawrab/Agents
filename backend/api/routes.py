from fastapi import APIRouter
from typing import List
from pydantic import BaseModel
from fastapi import UploadFile, File, HTTPException
import os
import traceback
from virtualgraph import run_graph
from virtualgraph_v2 import run_graph_v2
from web_scraper.scraper import scrape_remote
import logging
import json


logger = logging.getLogger(__name__)

router = APIRouter()


UPLOAD_FOLDER = 'csv_test'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class InitialState(BaseModel):
    csv_files: List[str]

@router.get("/")
async def index():
    return "Hello, World!"

        
RETURN_FILE = os.path.join(os.path.dirname(__file__), "return.txt")

with open(RETURN_FILE, "r", encoding="utf-8") as file:
    return_text = json.load(file)


@router.post("/upload-and-process")
async def test_frontend():
    return return_text


# @router.post("/upload-and-process")
# async def upload_and_process(files: List[UploadFile] = File(...)):
#     try:
#         logger.debug("Received upload request")
        
#         # Save uploaded files
#         logger.debug(f"Received {len(files)} files")
        
#         csv_files = []
        
#         for file in files:
#             if file.filename.endswith('.csv'):
#                 file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#                 logger.debug(f"Saving file to {file_path}")
#                 with open(file_path, "wb") as buffer:
#                     content = await file.read()
#                     buffer.write(content)
#                 csv_files.append(file_path)

#         # Execute the graph with the initial state
#         logger.debug("Starting graph execution")
#         result = run_graph(csv_files)
#         logger.debug("Graph execution completed")

#         return {
#             'status': 'success',
#             'message': 'Files processed successfully',
#             'result': result
#         }

#     except Exception as e:
#         logger.error(f"Error processing request: {str(e)}")
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         raise HTTPException(
#             status_code=500,
#             detail={
#                 'status': 'error',
#                 'message': str(e),
#                 'traceback': traceback.format_exc()
#             }
#         )


@router.post("/upload-and-process-v2")
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
        
        



