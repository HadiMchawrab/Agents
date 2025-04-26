import sys
import os
from pathlib import Path
import base64
# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from web_scraper.scraper import scrape
import pandas as pd
import json
import logging
from dotenv import load_dotenv
import asyncio


# Load environment variables
if not load_dotenv():
    logging.warning("Failed to load .env file. Ensure it exists and is properly configured.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    raise ValueError("CLAUDE_API_KEY is not set. Please check your .env file.")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
model = ChatAnthropic(model_name="claude-3-7-sonnet-20250219", temperature=0, anthropic_api_key=CLAUDE_API_KEY, max_tokens = 8192 )
model_GPT  = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"), max_tokens = 4052)

class State(TypedDict):
    tables: set  # set of dictionaries {'table1': ['col1', 'col2'], 'table2': ['col1', 'col2']}
    adjusted_columns: set # set of dictionaries {'table1': ['col1:datatype', 'col2:datatype'], 'table2': ['col1:datatype', 'col2:datatype']}
    data_frames: set # set of data frames created from the tables
    csv_files: set # initial set of csv files, we need them to create the data frames from tables
    topic: str # topic of the conversation 
    Relationship: str #the relationship between the tables columns, ML Models and the topic at hand
    ML_Models: set  
    DF_Info: set 
    Analysis: set 
    Pictures: set 
    Pictures_Analysis: set
    Reqs : set
    scripts: set # set of dictionaries of data frames and the scripts to run on them
    executed_notebook: set # the executed notebook after running the scripts on the data frames
    chosen_models : set # the chosen models after running the analysis on the images
    explained_models : set
    Last_Analysis: str # the analysis of the last model and the last data frame after running the analysis on the images
    Last_Model: str # the last model after running the analysis on the images
    Last_DF: str # the last data frame after running the analysis on the images
    FinalReqs: set # the final requirements after running the training scripts on the data frames
    FinalScripts: set # the final scripts after running the training scripts on the data frames





graph_builder = StateGraph(State)


def into_data_frames(state: State) -> State:
    # state['tables'] is a set of dictionaries {'table1': ['col1', 'col2'], 'table2': ['col1', 'col2']} 
    # hence we need to loop over the set, and for each table, make it a data frame and store it in state['data_frames']
    data_frames = {}
    adjusted_columns = {} # I want to store the adjusted columns in a set of dictionaries {'table1': ['col1:datatype', 'col2:datatype'], 'table2': ['col1:datatype', 'col2:datatype']} hence we need to find the type of each column in each data frame and insert them into this variable
    for table in state['tables']: 
        for table_name, columns in table.items():
            # Create a DataFrame from the CSV file in csv_test directory
            csv_file = f"csv_test/{table_name}.csv"
            if os.path.exists(csv_file):
                try:
                    available_df = pd.read_csv(csv_file)
                    available_columns = available_df.columns.tolist()
                    valid_columns = [col for col in columns if col in available_columns]
                    if len(valid_columns) < len(columns):
                        missing_columns = [col for col in columns if col not in available_columns]
                        logging.warning(f"Some requested columns for {table_name} don't exist in the CSV: {missing_columns}")
                    if valid_columns:
                        df = pd.read_csv(csv_file, usecols=valid_columns)
                        data_frames[table_name] = df
                        column_dtypes = []
                        for col in valid_columns:
                            dtype_str = str(df[col].dtype)
                            column_dtypes.append(f"{col}:{dtype_str}")
                        adjusted_columns[table_name] = column_dtypes
                    else:
                        logging.warning(f"None of the requested columns for {table_name} exist in the CSV.")
                except Exception as e:
                    logging.error(f"Error reading {csv_file}: {str(e)}")
            else:
                logging.warning(f"CSV file {csv_file} does not exist. Skipping this table.")
    logging.info(f"Data frames created: {adjusted_columns}")
    return {'data_frames': data_frames, 'adjusted_columns': adjusted_columns}



def generate_analysis(state: State) -> State:
    ans = {}
    Reqs = {}
    scripts = {}
    
    # Extract table names from adjusted_columns dictionary
    table_names = list(state['adjusted_columns'].keys())
    logging.info(f"Tables for analysis: {table_names}")
    
    for table_name in table_names:
        # Generate analysis for each DataFrame using CLAUDE
        logging.info(f"Generating analysis for {table_name}")
        adjusted_columns = state['adjusted_columns'][table_name]
        adjusted_columns_str = str(adjusted_columns)
        ml_models_str = str(state['ML_Models'])
        logging.info(f"Generating analysis for {table_name} with columns: {adjusted_columns_str} and ML models: {ml_models_str}")
        input_messages= [SystemMessage(content = """
                                                You will be generating python scripts to run on the data frame to generate the analysis and get visualization.
                                                I will then be sending the data you generated with the Data frames in a jupyter notebook to run the scripts and generate the analysis and get visualization.
                                                The scripts will be run in a single code cell, I dont want you to gererate big ammounts of code, just the scripts to run on the data frame to generate the analysis and get visualization and not to run models, limit the scripts to 200 lines of code max
                                                As a result of the scripts, I need to get pictures such as relationships between the columns, heat maps and so on.
                                                """), 
                     HumanMessage(content = f"""Return the response *only* in this strict JSON format, with no additional text or explanations:
                                ``` json
                                    {'{'}
                                        "Reqs": "All the requirements to be installed to run the below scripts seperated by a single space between each requirement, and the requirements should be in a single line",
                                        "Scripts": "You will continue working assuming that there is pandas dataframe called {table_name} which is defined to have these columns {adjusted_columns_str}. 
                                        You shouldn't re-write the before lines continue from there. 
                                        Your goal: generate a full working code, knowing that you need to generate images and analysis and give a set of visualizations of this current dataframe (which is given, so you shouldn't write it in your code, assume you have a dataframe called {table_name} ) we need to make analysis of the data to understand more how we can use ML models to tackle this topic {state['topic']}. Your goal is not to write an ML algorithm but just to provided illustraiotns analysis, distributions, heatmpas, boxplots, correlation table ... on the dataframe called: {table_name}.
                                        The scripts to run on the data frame to generate the analysis and get a set of visualizations not to train the models on our dataset, but rather to get visualizations on the data we have which would be relevant to use later when we want to choose the best ML Model , you can also either use the whole data frame or choose a subset of the columns (limit the scripts to 200 lines of code max, which will be running in a single code cell) and call the dataframe "{table_name}", exactly as it is. 

                                        IF you want to show the correlation table, you dont need columns that are strings, you should look at the columns that have type either int or float using this info {adjusted_columns_str}, and you can use the correlation table.                                        
                                        Also make use of other relationships between the columns and the ML models to generate the analysis and get visualizations, while taking into consideration the type of the data you are using in order not to get an error. 

                                        Now the path that you will be saving the pictures in is 'notebook_output/{table_name}/' and the name of the picture should be '{table_name}_figure_1.png' and so on, and the path should be relative to the notebook that will be running the scripts.
                                        

                                        Rules: 
                                        Inside your working directory which is notebook_output, 
                                        You need to create a subdirectory for this table name = {table_name} in the notebook_output directory, and save the pictures in it. 
                                        Example of the path: {table_name}/{table_name}_figure_1.png
                                                             {table_name}/{table_name}_figure_2.png
                                        And so on. "
                                    {'}'}
                                    ```
                                    YOU NEED TO RETURN THE JSON RESPONSE ONLY, STARTING with json and ending with , so I can parse it easily
                                    """),
                     HumanMessage(content = f"""The topic is: {state['topic']}"""),
                     HumanMessage(content = f"""The columns in this data frame are: {adjusted_columns_str}"""),
                     HumanMessage(content = f"""The ML models are: {ml_models_str}""")
                     ]
                                  

        
        logging.info(f"Message Sent to AI")
        ai_message = model_GPT.invoke(input_messages)
        logging.info(ai_message.content)
        raw_content = ai_message.content.strip()
        logging.info(f"Claude raw response: {raw_content}")

        json_text = None
        if raw_content.startswith("json"):
            start_index = raw_content.find("json") + len("json")
            end_index = raw_content.find("", start_index)
            json_text = raw_content[start_index:end_index].strip()
        elif raw_content.startswith(""):
            json_text = raw_content.strip("").strip()
        else:
            json_text = raw_content  # Try parsing whatever we get
        try:
            parsed_json = json.loads(json_text[7:-3])
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Claude response:\n{json_text}")
            raise ValueError("Claude's response was not valid JSON.") from e
            
        # Store results using string table_name as dictionary key
        Reqs[table_name] = parsed_json.get("Reqs", "No Reqs returned")
        scripts[table_name] = parsed_json.get("Scripts", "No Scripts returned")        
        
    return {'Analysis': ans, 'Reqs': Reqs, 'scripts': scripts}

import httpx

async def send_to_notebook(reqs: dict, scripts: dict, dfs: dict):
    logger = logging.getLogger(__name__)
    logger.info("Starting send_to_notebook function")
    logger.debug(f"Received reqs keys: {list(reqs.keys())}")
    logger.debug(f"Received scripts keys: {list(scripts.keys())}")
    logger.debug(f"Received dataframes keys: {list(dfs.keys())}")

    # Since we're running in Docker and the services are in the same network,
    # we can use localhost because the backend container has its own network namespace
    notebook_url = 'http://localhost:7000'
    
    csvs = []
    file_handles = []
    try:
        # Create csv_adjusted directory if it doesn't exist
        os.makedirs("csv_adjusted", exist_ok=True)
        logger.debug("Created csv_adjusted directory")
        
        files = []
        # Convert DataFrames to CSV files and prepare files dict
        for table_name, df in dfs.items():
            logger.info(f"Processing table: {table_name}")
            csv_file = f"csv_adjusted/{table_name}.csv"  # Include .csv extension
            logger.debug(f"Saving DataFrame to {csv_file}")
            
            try:
                df.to_csv(csv_file, index=False)
                csvs.append(csv_file)
                logger.debug(f"Successfully saved DataFrame to {csv_file}")
            except Exception as e:
                logger.error(f"Error saving DataFrame to CSV: {str(e)}")
                raise
            
            try:
                # Open file and create file handle
                file_handle = open(csv_file, 'rb')
                file_handles.append(file_handle)
                # Add to files list in the correct format for httpx
                files.append(('files', (f'file_{table_name}', file_handle, 'text/csv')))
                logger.debug(f"Created file handle and added to files list for {table_name}")
            except Exception as e:
                logger.error(f"Error creating file handle: {str(e)}")
                raise

        # Prepare the data as JSON strings
        try:
            data = {
                'reqs': (None, json.dumps(reqs)),
                'scripts': (None, json.dumps(scripts))
            }
            logger.debug("Successfully prepared JSON data")
        except Exception as e:
            logger.error(f"Error preparing JSON data: {str(e)}")
            raise
        
        # Use async context manager with increased timeouts
        timeout_settings = httpx.Timeout(
            timeout=600.0,  # 10 minutes for the entire operation
            connect=60.0,   # 1 minute for connecting
            read=600.0,     # 10 minutes for reading
            write=60.0      # 1 minute for writing
        )
        
        logger.info("Preparing to send request to notebook service")
        async with httpx.AsyncClient(timeout=timeout_settings) as client:
            try:
                logger.debug(f"Sending POST request to notebook service with {len(files)} files")
                response = await client.post(
                    notebook_url + "/analyze-data",
                    data=data,
                    files=files
                )
                logger.debug(f"Received response with status code: {response.status_code}")
                
                if response.status_code == 422:
                    logger.error(f"Validation error response: {response.text}")
                    return {"error": f"Request validation failed: {response.text}"}
                
                response_json = response.json()
                logger.info("Successfully received and parsed response from notebook service")
                return response_json
            except httpx.TimeoutException as e:
                logger.error(f"Timeout error while sending data to notebook service: {str(e)}")
                return {"error": "Request timed out while sending data to notebook service"}
            except httpx.RequestError as e:
                logger.error(f"Error sending request to notebook service: {str(e)}")
                return {"error": f"Failed to send request to notebook service: {str(e)}"}
            except Exception as e:
                logger.error(f"Unexpected error while communicating with notebook service: {str(e)}", exc_info=True)
                return {"error": f"Unexpected error: {str(e)}"}
    
    finally:
        logger.debug("Starting cleanup process")
        # Close all file handles
        for handle in file_handles:
            try:
                handle.close()
                logger.debug("Closed file handle")
            except Exception as e:
                logger.error(f"Error closing file handle: {str(e)}")
        
        # Clean up temporary CSV files after handles are closed
        for csv_file in csvs:
            try:
                if os.path.exists(csv_file):
                    os.remove(csv_file)
                    logger.debug(f"Removed temporary file: {csv_file}")
            except Exception as e:
                logger.error(f"Error cleaning up {csv_file}: {str(e)}")

def call_notebook_service(state: State) -> State:
    """
    Async node to call the notebook FastAPI service and return the executed notebook.
    """
    reqs = state.get("Reqs", {})
    scripts = state.get("scripts", {})
    dfs = state.get("data_frames", {})
    
    notebook_result = asyncio.run(send_to_notebook(reqs, scripts, dfs))
    state["executed_notebook"] = notebook_result
    return state


def analyze_images(state: State) -> State:
    ml_mod = {}
    model_analysis = {}
    Last_Analysis = str
    Last_Model = str
    Last_DF = str

    pictures = state['executed_notebook']['images']

    # pictures = {
    #     "banking": [
    #         "notebook_output/banking/banking_figure_1.png",
    #         "notebook_output/banking/banking_figure_2.png",
    #         "notebook_output/banking/banking_figure_3.png",
    #         "notebook_output/banking/banking_figure_4.png",
    #         "notebook_output/banking/banking_figure_5.png",
    #     ],
    #     "data": [
    #         "notebook_output/data/data_figure_1.png",
    #         "notebook_output/data/data_figure_2.png",
    #         "notebook_output/data/data_figure_3.png",
    #         "notebook_output/data/data_figure_4.png",
    #         "notebook_output/data/data_figure_5.png",
    #         "notebook_output/data/data_figure_6.png",
    #         "notebook_output/data/data_figure_7.png",
    #         "notebook_output/data/data_figure_8.png",
    #         "notebook_output/data/data_figure_9.png",
    #     ],
    # }

    state["Pictures_Analysis"] = {}
    for table, paths in pictures.items():
        image_messages = []
        for png_path in paths:
            with open(png_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            image_messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

        # 2) Compose the chat messages
        messages = [
            SystemMessage(content=(
                "You are an expert data scientist. "
                "Analyze the set of exploratory plots provided and extract key insights "
                "about data distribution, feature relationships, and any anomalies. "
                "Then recommend the single best ML model for this dataset."
            )),
            HumanMessage(content = f"""Return the response *only* in this strict JSON format, with no additional text or explanations:     
                        ```json
                                    {'{'}
                                        "RecommendedModel": "one of the set of ML models: {state['ML_Models']}, and the model should be in a single line",
                                        "Insights": "Why we chose this model based on the analysis of the images",
                                        
                                    {'}'}```"""),
            HumanMessage(content=image_messages)
        ]

        logging.info(f"Message Sent to AI")
        ai_message = model_GPT.invoke(messages)
        logging.info(ai_message.content)
        raw_content = ai_message.content.strip()
        logging.info(f"Claude raw response: {raw_content}")

        json_text = None
        if raw_content.startswith("json"):
            start_index = raw_content.find("json") + len("json")
            end_index = raw_content.find("", start_index)
            json_text = raw_content[start_index:end_index].strip()
        elif raw_content.startswith(""):
            json_text = raw_content.strip("").strip()
        else:
            json_text = raw_content  # Try parsing whatever we get
        try:
            parsed_json = json.loads(json_text[7:-3])
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Claude response:\n{json_text}")
            raise ValueError("Claude's response was not valid JSON.") from e

        ml_mod[table] = parsed_json.get("RecommendedModel", "No Insights returned")
        model_analysis[table] = parsed_json.get("Insights", "No Model returned")
    logging.info(f"Model Analysis: {model_analysis}")
    logging.info(f"ML Models: {ml_mod}")
    stringified_models_analysis = str(state['explained_models'])
    stringified_models = str(state['chosen_models'])

    messages = [
    SystemMessage(content=f"""(
        "You are an expert data scientist. "
        "Analyze the dataframes within the variables given the explanation {stringified_models_analysis}on why each ML model is being used in each dataframe"
        "From these inferences, check which is the most understandable in why we chose the model  "
        "Then recommend the single best data frame and its compatible ML model."
    )"""),
    HumanMessage(content = f"""Return the response *only* in this strict JSON format, with no additional text or explanations:     
                ```json
                            {'{'}
                                "Last_DF": "one of the set of tables models: {state['tables']}",
                                "Last_Model":"model from {stringified_models} should be in a single line",
                                "Last_Analysis": "the analysis of the last model and the last data frame",
                                
                            {'}'}```""")
        ]

    logging.info(f"Message Sent to AI")
    ai_message2 = model_GPT.invoke(messages)
    logging.info(ai_message2.content)
    raw_content2 = ai_message2.content.strip()
    logging.info(f"Claude raw response: {raw_content2}")

    json_text = None
    if raw_content2.startswith("json"):
        start_index = raw_content2.find("json") + len("json")
        end_index = raw_content2.find("", start_index)
        json_text = raw_content2[start_index:end_index].strip()
    elif raw_content2.startswith(""):
        json_text = raw_content2.strip("").strip()
    else:
        json_text = raw_content2  # Try parsing whatever we get
    try:
        parsed_json2 = json.loads(json_text[7:-3])
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse Claude response:\n{json_text}")
        raise ValueError("Claude's response was not valid JSON.") from e

    Last_DF = parsed_json2.get("Last_DF", "No Last DF returned")
    Last_Model = parsed_json2.get("Last_Model", "No Last Model returned")
    Last_Analysis = parsed_json2.get("Last_Analysis", "No Last Analysis returned")



    return {'chosen_models': Last_Model, 'explained_models': Last_Analysis, 'Last_DF': Last_DF}






def generate_train(state: State) -> State:
    stringified_analysis = str(state['Analysis'])
    stringified_model = str(state['chosen_models'])

    messages = [
    SystemMessage(content=("You are an expert data scientist. "
        "Given the ML model to use and the analysis and the data frame, generate a python script that would train the model on the data frame."
       
    )),
    HumanMessage(content = f"""Return the response *only* in this strict JSON format, with no additional text or explanations:     
                ```json
                            {'{'},
                                "LRequiremenets": "All the requirements to be installed to run the below scripts seperated by a single space between each requirement, and the requirements should be in a single line",
                                "LScripts": "Generate a full python text, which would run the dataframe {state['Last_DF']} and train the model {stringified_model} on it, and the script should be in a single line and separated by \n. Also save ethe result of the training in a file called {state['Last_DF']}_model.pkl, and the path should be relative to the notebook that will be running the scripts.",
                                
                            {'}'}```""")
                            ]

    logging.info(f"Message Sent to AI")
    ai_message = model_GPT.invoke(messages)
    logging.info(f"Message Sent to AI")
    ai_message2 = model_GPT.invoke(messages)
    logging.info(ai_message2.content)
    raw_content2 = ai_message2.content.strip()
    logging.info(f"Claude raw response:final {raw_content2}")

    json_text = None
    if raw_content2.startswith("json"):
        start_index = raw_content2.find("json") + len("json")
        end_index = raw_content2.find("", start_index)
        json_text = raw_content2[start_index:end_index].strip()
    elif raw_content2.startswith(""):
        json_text = raw_content2.strip("").strip()
    else:
        json_text = raw_content2  # Try parsing whatever we get
    try:
        parsed_json3 = json.loads(json_text[7:-3])
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse Claude response:\n{json_text}")
        raise ValueError("Claude's response was not valid JSON.") from e
    logging.info(f"Claude raw response: {parsed_json3}")
    Final_Scripts = parsed_json3.get("LScripts", "No Last DF returned")
    Final_Reqs = parsed_json3.get("LRequiremenets", "No Last Model returned")

    return {'FinalReqs': Final_Reqs, 'FinalScripts': Final_Scripts}




graph_builder.add_node(into_data_frames, "into_data_frames")
graph_builder.add_node(generate_analysis, "generate_analysis")
graph_builder.add_node(call_notebook_service, "call_notebook_service")
graph_builder.add_node(analyze_images, "analyze_images")
graph_builder.add_node(generate_train, "generate_train")

graph_builder.add_edge("into_data_frames", "generate_analysis")
graph_builder.add_edge("generate_analysis", "call_notebook_service")
graph_builder.add_edge("call_notebook_service", "analyze_images")
graph_builder.add_edge("analyze_images", "generate_train")
graph_builder.set_entry_point("into_data_frames")
graph_builder.set_finish_point("generate_train")

graph2 = graph_builder.compile()


def test_graph2():
    # Create a sample state
    initial_state = {
        'tables': [{'banking': ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'duration']}, {'data': ['Bankrupt?', ' ROA(C) before interest and depreciation before interest', ' Debt ratio %', ' Net worth/Assets', ' Total debt/Total net worth', ' Cash Flow to Liability', ' Net Income to Total Assets']}],
        'adjusted_columns': {},
        'data_frames': {},
        'csv_files': {'csv_test/banking.csv', 'csv_test/data.csv'},
        'topic': "Machine learning models employed in Credit Risk Assessment",
        'Relationship': "The ML models (SVM, RNN, Bayesian Networks) are well-suited for credit risk assessment using the available data. SVM can classify customers into risk categories based on financial attributes, RNNs can analyze sequential patterns in financial data to predict defaults, and Bayesian Networks can model probabilistic relationships between financial indicators and credit risk outcomes.",
        'ML_Models': ["Support Vector Machines (SVM)", "Recurrent Neural Networks (RNN)", "Bayesian Networks","Logistic Regression", "Random Forest", "Gradient Boosting", "Neural Networks"],
        'scripts': {}, # set of dictionaries of data frames and the scripts to run on them
        'Analysis': {},
        'Pictures': {},
        'Pictures_Analysis': {},
        'Reqs': {},
        'DF_Info': {},
        'executed_notebook': {},
        'chosen_models': {},
        'explained_models': {},
        'FinalReqs': {},
        'FinalScripts': {},
        'Last_Analysis': {},
        'Last_Model': {},
        'Last_DF': {}


    }

    # Run the graph with the sample state
    print("Initial State:")
    final_state2 = graph2.invoke(initial_state)
    print(final_state2)
    
def run_graph2(data: dict) -> State:
    # Transform tables from {table: columns} to [{table: columns}]
    tables_list = [
        {table_name: columns} 
        for table_name, columns in data['tables'].items()
    ]
    print(tables_list)
    
    initial_state = {
        'tables': tables_list,
        'adjusted_columns': {},
        'data_frames': {},
        'csv_files': data['csv_files'],
        'topic': data['topic'],
        'Relationship': data['Relationship'],
        'ML_Models': data['ML_Models'],
        'DF_Info':{},
        'Analysis':{},
        'Pictures':{},
        'Pictures_Analysis':{},
        'Reqs': {},
        'scripts': {}, 
        'executed_notebook': {},
        'chosen_models': {},
        'explained_models': {},
        'FinalReqs': {},
        'FinalScripts': {},
        'Last_Analysis': {},
        'Last_Model': {},
        'Last_DF': {}

        }
    
    print(initial_state)
    final_state2 = graph2.invoke(initial_state)
    print(final_state2)
    
    return {
        'tables': final_state2.get('tables', []),
        'adjusted_columns': final_state2.get('adjusted_columns', {}),
        'csv_files': list(final_state2.get('csv_files', [])),
        'topic': final_state2.get('topic', ''),
        'Relationship': final_state2.get('Relationship', ''),
        'ML_Models': final_state2.get('ML_Models', []),
        'Analysis': final_state2.get('Analysis', {}),
        'DF_Info': final_state2.get('DF_Info', {}),
        'Pictures': final_state2.get('Pictures', {}),
        'Pictures_Analysis': final_state2.get('Pictures_Analysis', {}),
        'Reqs': final_state2.get('Reqs', {}),
        'scripts': final_state2.get('scripts', {}),
        'executed_notebook': final_state2.get('executed_notebook', {}),
        'chosen_models': final_state2.get('chosen_models', {}),
        'explained_models': final_state2.get('explained_models', {}),
        'FinalReqs': final_state2.get('FinalReqs', {}),
        'FinalScripts': final_state2.get('FinalScripts', {}),
        'Last_Analysis': final_state2.get('Last_Analysis', ''),
        'Last_Model': final_state2.get('Last_Model', ''),
        'Last_DF': final_state2.get('Last_DF', '')




    }
    

    # return {
    #     'tables': final_state2.get('tables', []),
    #     'adjusted_columns': final_state2.get('adjusted_columns', {}),
    #     'csv_files': list(final_state2.get('csv_files', [])),
    #     'topic': final_state2.get('topic', ''),
    #     'Relationship': final_state2.get('Relationship', ''),
    #     'ML_Models': final_state2.get('ML_Models', []),
    #     'Analysis': final_state2.get('Analysis', {})
    # }


if __name__ == "__main__":
    test_graph2()