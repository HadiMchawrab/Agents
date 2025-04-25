from fastapi import APIRouter, Request, UploadFile, File, BackgroundTasks, HTTPException, Form
from typing import List, Dict, Any
from pydantic import BaseModel
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
import subprocess
import json
import pandas as pd
import asyncio
import logging
from contextlib import contextmanager
import os
import textwrap


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

router = APIRouter()

@contextmanager
def cleanup_files(*files):
    try:
        yield
    finally:
        for file in files:
            try:
                if os.path.exists(file):
                    os.remove(file)
                    logger.debug(f"Cleaned up file: {file}")
            except Exception as e:
                logger.error(f"Error cleaning up {file}: {str(e)}")

class AnalysisRequest(BaseModel):
    reqs: str
    scripts: str

@router.post("/analyze-data")
async def data_analysis(
    reqs: str = Form(...),
    scripts: str = Form(...),
    files: List[UploadFile] = File(default=...)
):
    logger.info("Starting data analysis request")
    logger.debug(f"Received requirements: {reqs}")
    logger.debug(f"Received scripts: {scripts}")
    logger.debug(f"Received files: {[f.filename for f in files]}")

    temp_files = []
    notebook_filename = 'temp_notebook.ipynb'
    executed_notebook_filename = 'executed_notebook.ipynb'
    output_dir = 'notebook_output'
    
    try:
        # Create output directory for figures
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Created output directory: {output_dir}")
        
        # Parse the JSON strings into Python dictionaries
        logger.debug("Attempting to parse JSON data")
        try:
            reqs_dict = json.loads(reqs)
            scripts_dict = json.loads(scripts)
            logger.debug(f"Successfully parsed JSON. Requirements keys: {list(reqs_dict.keys())}")
            logger.debug(f"Scripts keys: {list(scripts_dict.keys())}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
        
        if not files:
            logger.error("No files were provided in the request")
            raise HTTPException(status_code=400, detail="No files were provided")
            
        dfs = {}
        # Create temporary directory for CSV files
        logger.debug("Creating temporary directory for CSV files")
        os.makedirs("temp_csv", exist_ok=True)
        
        for file in files:
            logger.info(f"Processing file: {file.filename}")
            # Extract the original table name from the file key (remove 'file_' prefix)
            table_name = file.filename
            if table_name.startswith('file_'):
                table_name = table_name[5:]  # Remove 'file_' prefix
                logger.debug(f"Extracted table name: {table_name} from filename: {file.filename}")
            
            try:
                contents = await file.read()
                logger.debug(f"Successfully read contents of file: {file.filename}")
            except Exception as e:
                logger.error(f"Error reading file {file.filename}: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error reading file {file.filename}: {str(e)}")

            # Save CSV to temporary file
            temp_csv = f"temp_csv/{table_name}.csv"
            logger.debug(f"Saving contents to temporary file: {temp_csv}")
            try:
                with open(temp_csv, "wb") as f:
                    f.write(contents)
                temp_files.append(temp_csv)
                logger.debug(f"Successfully saved file: {temp_csv}")
            except Exception as e:
                logger.error(f"Error saving temporary file {temp_csv}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error saving temporary file: {str(e)}")

            # Read the CSV file
            try:
                logger.debug(f"Attempting to read CSV file: {temp_csv}")
                dfs[table_name] = pd.read_csv(temp_csv)
                logger.debug(f"Successfully loaded DataFrame for {table_name} with shape {dfs[table_name].shape}")
            except Exception as e:
                logger.error(f"Error reading CSV file {table_name}: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error reading CSV file {table_name}: {str(e)}")
            
        if not dfs:
            logger.error("No valid CSV files were processed")
            raise HTTPException(status_code=400, detail="No valid CSV files were processed")
            
        for tablename in dfs.keys():    
            logger.info(f"Processing table: {tablename}")
            if tablename not in reqs_dict:
                logger.error(f"No requirements found for table {tablename}")
                raise HTTPException(status_code=400, detail=f"No requirements found for table {tablename}")
            if tablename not in scripts_dict:
                logger.error(f"No script found for table {tablename}")
                raise HTTPException(status_code=400, detail=f"No script found for table {tablename}")
                
            requirements = reqs_dict[tablename]
            script = scripts_dict[tablename]
            logger.debug(f"Requirements for {tablename}: {requirements}")
            logger.debug(f"Script length for {tablename}: {len(script)}")
            
            # Create a new notebook
            logger.debug("Creating new notebook")
            nb = new_notebook()

            # Add cells to notebook
            logger.debug("Adding cells to notebook")
            # First cell: Install requirements
            install_cell = new_code_cell(f"""%%time
!pip install {requirements}""")
            nb.cells.append(install_cell)
            
            # Second cell: Import statements and setup
            setup_cell = new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

%matplotlib inline
plt.style.use('seaborn-v0_8')

# Create output directory for figures
os.makedirs('notebook_output', exist_ok=True)""")
            nb.cells.append(setup_cell)
            
            # Third cell: Load df

            df_code = textwrap.dedent(f"""
                # Load and preview the df
                {tablename} = pd.read_csv('temp_csv/{tablename}.csv')
                print("data shape:", {tablename}.shape)
                print("\\nFirst few rows:")
                print({tablename}.head())
                print("Columns:", {tablename}.columns.tolist())
            """)
            
            df_cell = new_code_cell(df_code)

            nb.cells.append(df_cell)

            # Fourth cell: Modify the script to use the output directory
            modified_script = script.replace("plt.savefig('", "plt.savefig('notebook_output/")
            script_cell = new_code_cell(modified_script)
            nb.cells.append(script_cell)

            # Write the notebook to a file
            logger.debug(f"Writing notebook to file: {notebook_filename}")
            try:
                with open(notebook_filename, 'w') as f:
                    nbformat.write(nb, f)
                temp_files.append(notebook_filename)
                logger.debug(f"Successfully wrote notebook to {notebook_filename}")
            except Exception as e:
                logger.error(f"Error writing notebook file: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error writing notebook file: {str(e)}")

            # Execute the notebook using synchronous subprocess
            logger.info("Executing notebook")
            try:
                import subprocess
                
                cmd = [
                    'jupyter', 'nbconvert', 
                    '--to', 'notebook',
                    '--execute',
                    '--ExecutePreprocessor.timeout=600',  # Increased to 10 minutes
                    notebook_filename,
                    '--output', executed_notebook_filename
                ]
                
                logger.debug(f"Running command: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=600  # Added 10 minute timeout for the subprocess itself
                )
                
                stdout_text = result.stdout
                stderr_text = result.stderr
                
                logger.debug(f"Notebook execution stdout:\n{stdout_text}")
                if stderr_text:
                    logger.error(f"Notebook execution stderr:\n{stderr_text}")
                
                if result.returncode != 0:
                    error_msg = stderr_text or "Unknown error"
                    logger.error(f"Notebook execution failed with return code {result.returncode}. Error: {error_msg}")
                    raise HTTPException(status_code=500, detail=f"Notebook execution failed: {error_msg}")

                temp_files.append(executed_notebook_filename)
                logger.info("Notebook execution completed successfully")

                # Get list of generated figures
                figures = []
                if os.path.exists(output_dir):
                    figures = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.pdf'))]
                logger.debug(f"Generated figures: {figures}")

                # Read and return the executed notebook along with figure paths
                logger.debug(f"Reading executed notebook: {executed_notebook_filename}")
                with open(executed_notebook_filename) as f:
                    executed_nb = f.read()
                
                # return {
                #     "executed_notebook": executed_nb,
                #     "figures": [os.path.join(output_dir, f) for f in figures]
                # }

            except Exception as e:
                logger.error(f"Error during notebook execution: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error during notebook execution: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in data analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    # finally:
    #     # Don't clean up the output directory since we need the figures
    #     # But clean up temporary files
    #     logger.debug("Starting cleanup of temporary files")
    #     for file in temp_files:
    #         try:
    #             if os.path.exists(file):
    #                 os.remove(file)
    #                 logger.debug(f"Cleaned up file: {file}")
    #         except Exception as e:
    #             logger.error(f"Error cleaning up {file}: {str(e)}")
    #     # Clean up temp directory
    #     try:
    #         if os.path.exists("temp_csv"):
    #             os.rmdir("temp_csv")
    #             logger.debug("Cleaned up temp_csv directory")
    #     except Exception as e:
    #         logger.error(f"Error cleaning up temp_csv directory: {str(e)}")
