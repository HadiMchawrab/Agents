from fastapi import APIRouter, HTTPException, status, Request
from typing import List, Dict
from pydantic import BaseModel
from fastapi import UploadFile, File
import os
import traceback
from virtualgraph import run_graph, test
from web_scraper.scraper import scrape_remote
import logging
import json
from implementer import run_graph2

logger = logging.getLogger(__name__)

router = APIRouter()


UPLOAD_FOLDER = 'csv_test'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


class SubmitDataRequest(BaseModel):
    tables: Dict[str, List[str]]
    topic: str
    Relationship: List[str]
    ML_Models: List[str]

@router.post("/scraper-test")
async def scraper_test():
    return scrape_remote("Machine learning models employed in credit risk assessment and bankruptcy prediction")

@router.post("/test")
async def test_endpoint():
    await test()
    
def convert_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {k: convert_sets(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_sets(item) for item in obj]
    return obj

def convert_numpy_types(obj):
    """Convert numpy types and pandas objects to Python native types for JSON serialization"""
    import numpy as np
    import pandas as pd
    
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, set):
        return list(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        return obj.to_list()
    else:
        return obj

@router.post("/upload-and-process")
async def upload_and_process(request: Request, files: List[UploadFile] = File(...)):
    try:
        
        logger.debug("Received upload request")
        
        form = await request.form()
        descriptions = {}

        for key, value in form.items():
            if key.startswith("descriptions["):
                filename = key[len("descriptions["):-1]
                descriptions[filename] = value
        print(descriptions)
        
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
            result = run_graph(csv_files, descriptions)
            logger.debug("Graph execution completed")
            
            # Save the result to a JSON file
            output_dir = "graph_results"
            os.makedirs(output_dir, exist_ok=True)
            full_path = os.path.join(output_dir, "graph_results.json")
            
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(convert_sets(result), f, indent=4, ensure_ascii=False)
            
            logger.info(f"Graph results saved to {full_path}")

            return {
                'status': 'success',
                'message': 'Files processed successfully',
                'result': convert_sets(result)
            }

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
async def submit_data(request_data: dict):
    try:
        logger.info("Received submit-data request")
        logger.debug(f"Request data: {request_data}")
        
        # Create the csv_files set with all required CSV files
        csv_files = set()
        for table_name in request_data.get('tables', {}):
            csv_file = f"csv_test/{table_name}.csv"
            if os.path.exists(csv_file):
                csv_files.add(csv_file)
            else:
                logger.error(f"CSV file not found: {csv_file}")
                raise HTTPException(status_code=404, detail=f"CSV file not found: {table_name}.csv")

        # Update request_data with csv_files
        request_data['csv_files'] = csv_files
        
        try:
            result = run_graph2(request_data)
            logger.info("Successfully processed submit-data request")
            return result
        except Exception as e:
            logger.error(f"Error processing submit-data request: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    'status': 'error',
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }
            )
    except Exception as e:
        logger.error(f"Error in submit-data endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                'status': 'error',
                'message': str(e),
                'traceback': traceback.format_exc()
            }
        )