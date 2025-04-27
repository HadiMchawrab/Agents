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
import base64
import glob


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
def data_analysis(
    reqs: str = Form(...),
    scripts: str = Form(...),
    files: List[UploadFile] = File(...)
):
    logger.info("Starting data analysis request")
    logger.debug(f"Received requirements: {reqs}")
    logger.debug(f"Received scripts: {scripts}")
    logger.debug(f"Received files: {[f.filename for f in files]}")

    temp_files = []
    notebook_filename = 'temp_notebook.ipynb'
    executed_notebook_filename = 'executed_notebook.ipynb'
    output_dir = '/notebook_output'
    
    results = {}
    
    try:
        # Create base output directory with full permissions
        os.makedirs(output_dir, mode=0o777, exist_ok=True)
        logger.debug(f"Created base output directory: {output_dir}")
        
        # Parse the JSON strings into Python dictionaries
        logger.debug("Attempting to parse JSON data")
        try:
            reqs_dict = json.loads(reqs)
            scripts_dict = json.loads(scripts)
            logger.debug(f"Successfully parsed JSON. Requirements keys: {list(reqs_dict.keys())}")
            logger.debug(f"Scripts keys: {list(scripts_dict.keys())}")
            
            # Create table-specific directories with full permissions
            for table_name in scripts_dict.keys():
                table_dir = os.path.join(output_dir, table_name)
                os.makedirs(table_dir, mode=0o777, exist_ok=True)
                logger.debug(f"Created table-specific directory: {table_dir}")

                # Fix the script to use correct path
                modified_script = scripts_dict[table_name]
                modified_script = modified_script.replace(
                    f"'{table_name}/{table_name}_figure_",
                    f"'/notebook_output/{table_name}/{table_name}_figure_"
                )
                modified_script = modified_script.replace(
                    f"'/notebook_output/{table_name}/notebook_output/{table_name}/",
                    f"'/notebook_output/{table_name}/"
                )
                scripts_dict[table_name] = modified_script
                logger.debug(f"Updated script paths for {table_name}")

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
                # Synchronously read file contents
                contents = file.file.read()
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

# Create output directory for figures in shared volume
os.makedirs('/notebook_output', exist_ok=True)""")
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

            # Fourth cell: Modify the script to use the shared volume output directory
            # Create subdirectory for each table in the output dir
            table_dir_cell = new_code_cell(f"""
            # Create subdirectory for {tablename} if it doesn't exist
            os.makedirs('/notebook_output/{tablename}', exist_ok=True)
            """)
            nb.cells.append(table_dir_cell)
            
            # Update the script to use the shared volume path
            modified_script = script.replace("'{tablename}/", "'/notebook_output/{tablename}/")
            modified_script = modified_script.replace("notebook_output/", "/notebook_output/")
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

            except Exception as e:
                logger.error(f"Error during notebook execution: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error during notebook execution: {str(e)}")
            
        # Collect and encode figures
        image_data = {}
        for table_name in dfs.keys():
            logger.debug(f"Collecting figures for table: {table_name}")
            image_files = glob.glob(f"/notebook_output/{table_name}/{table_name}_figure_*.png")
            logger.debug(f"Found {len(image_files)} image files for {table_name}")
            image_data[table_name] = []
            
            for image_file in image_files:
                try:
                    with open(image_file, "rb") as img_f:
                        encoded = base64.b64encode(img_f.read()).decode('utf-8')
                        image_data[table_name].append(encoded)
                        logger.debug(f"Successfully encoded image: {image_file}")
                except Exception as e:
                    logger.error(f"Error encoding image {image_file}: {str(e)}")
                    # Continue with other images even if one fails
                    continue

        # Return success response with images
        return {
            "message": "Analysis completed successfully",
            "status": "success",
            "images": image_data
        }
    
    except Exception as e:
        logger.error(f"Unexpected error in data analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))