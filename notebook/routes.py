from fastapi import APIRouter, Request, UploadFile, File
from typing import List, Dict
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
import subprocess
import json

router = APIRouter()

@router.post("/analyze-data")
async def data_analysis(reqs: set, scripts: set, dfs: set):
    for df in dfs:
        requirements = reqs[df].replace("\n", " ")
        script = scripts[df]
        
        # Create a new notebook
        nb = new_notebook()

        # Add a cell to install requirements
        install_cell = new_code_cell(f"!pip install {requirements}")
        nb.cells.append(install_cell)
        
        # Add a cell to load the dataframe
        df_cell = new_code_cell(f"import pandas as pd\ndf = pd.read_json('{json.dumps(df)}')")
        nb.cells.append(df_cell)

        # Add a cell with the script
        script_cell = new_code_cell(script)
        nb.cells.append(script_cell)

        # Write the notebook to a file
        notebook_filename = 'temp_notebook.ipynb'
        with open(notebook_filename, 'w') as f:
            nbformat.write(nb, f)

        # Execute the notebook
        subprocess.run([
            'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
            notebook_filename, '--output', 'executed_notebook.ipynb'
        ])

        # Read and return the executed notebook
        with open('executed_notebook.ipynb') as f:
            executed_nb = f.read()

        return {"executed_notebook": executed_nb}
    