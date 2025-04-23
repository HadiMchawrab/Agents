from fastapi import APIRouter, HTTPException, status
from typing import List
from pydantic import BaseModel
from fastapi import UploadFile, File
import os
import traceback
from virtualgraph import run_graph
from web_scraper.scraper import scrape_remote
import logging
import json
from anthropic import OverloadedError

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


# @router.post("/upload-and-process")
# async def test_frontend():
#     return return_text

@router.post("/scraper-test")
async def scraper_test():
    return scrape_remote("Machine learning models employed in credit risk assessment and bankruptcy prediction")


@router.post("/upload-and-process")
async def upload_and_process(files: List[UploadFile] = File(...)):
    try:
        logger.debug("Received upload request")
        
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No files were uploaded"
            )
        
        csv_files = []
        
        for file in files:
            if not file.filename.endswith('.csv'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File {file.filename} is not a CSV file"
                )
                
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            logger.debug(f"Saving file to {file_path}")
            
            try:
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                csv_files.append(file_path)
            except Exception as e:
                logger.error(f"Error saving file {file.filename}: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error saving file {file.filename}"
                )

        try:
            logger.debug("Starting graph execution")
            result = run_graph(csv_files)
            logger.debug("Graph execution completed")

            return {
                'status': 'success',
                'message': 'Files processed successfully',
                'result': result
            }
        except OverloadedError as e:
            logger.error(f"Anthropic API overloaded: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The AI service is currently overloaded. Please try again later."
            )
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    'status': 'error',
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )

        
        
@router.post("/submit-data")
async def submit_data(data: dict):
    return data


