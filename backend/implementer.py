import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from web_scraper.scraper import scrape
import pandas as pd
import json
import logging
from dotenv import load_dotenv


# Load environment variables
if not load_dotenv():
    logging.warning("Failed to load .env file. Ensure it exists and is properly configured.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    raise ValueError("CLAUDE_API_KEY is not set. Please check your .env file.")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
model = ChatAnthropic(model_name="claude-3-7-sonnet-20250219", temperature=0, anthropic_api_key=CLAUDE_API_KEY, max_tokens = 8192 )


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
    return_format: set



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
    return_format = {}
    
    # Extract table names from adjusted_columns dictionary
    table_names = list(state['adjusted_columns'].keys())
    logging.info(f"Tables for analysis: {table_names}")
    
    for table_name in table_names:
        # Generate analysis for each DataFrame using CLAUDE
        logging.info(f"Generating analysis for {table_name}")
        adjusted_columns_str = str(state['adjusted_columns'])
        ml_models_str = str(state['ML_Models'])
        logging.info(f"Generating analysis for {table_name} with columns: {adjusted_columns_str} and ML models: {ml_models_str}")
        input_messages= [SystemMessage(content = """
                                                You will be generating python scripts to run on the data frame to generate the analysis and get visualization.
                                                You will also need to generate the requirements to run the scripts.
                                                I will then be sending the data you generated with the Data frames in a jupyter notebook to run the scripts and generate the analysis and get visualization.
                                                The scripts will be run in a single code cell, I dont want you to gererate big ammounts of code, just the scripts to run on the data frame to generate the analysis and get visualization and not to run models, limit the scripts to 50 lines of code max
                                                As a result of the scripts, I need to get pictures such as relationships between the columns, heat maps and so on.
                                                """), 
                     HumanMessage(content = """"Return the response **only** in this strict JSON format, with no additional text or explanations:
                                 ```json
                                    {
                                        "Reqs": "All the requirements to be installed to run the below scripts",
                                        "Scripts": "The scripts to run on the data frame to generate the analysis and get a set of visualizations not to train the models on our dataset, but rather to get visualizations on the data we have which would be relevant to use later when we want to choose the best ML Model , you can also either use the whole data frame or choose a subset of the columns (limit the scripts to 50 lines of code max, which will be running in a single code cell)",
                                        "Return_Format": "The format of the return data, meaning how will you return "                                                                   
                                    }
                                    ```"""),
                     HumanMessage(content = f"""The topic is: {state['topic']}"""),
                     HumanMessage(content = f"""The columns in this data frame are: {adjusted_columns_str}"""),
                     HumanMessage(content = f"""The ML models are: {ml_models_str}""")
                     ]
                                  

        
        logging.info(f"Message Sent to AI")
        ai_message = model.invoke(input_messages)
        logging.info(ai_message.content)
        raw_content = ai_message.content.strip()
        logging.info(f"Claude raw response: {raw_content}")

        json_text = None
        if raw_content.startswith("```json"):
            start_index = raw_content.find("```json") + len("```json")
            end_index = raw_content.find("```", start_index)
            json_text = raw_content[start_index:end_index].strip()
        elif raw_content.startswith("```"):
            json_text = raw_content.strip("```").strip()
        else:
            json_text = raw_content  # Try parsing whatever we get
        try:
            parsed_json = json.loads(json_text)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Claude response:\n{json_text}")
            raise ValueError("Claude's response was not valid JSON.") from e
            
        # Store results using string table_name as dictionary key
        Reqs[table_name] = parsed_json.get("Reqs", "No Reqs returned")
        scripts[table_name] = parsed_json.get("Scripts", "No Scripts returned")
        return_format[table_name] = parsed_json.get("Return_Format", "No Format returned")
        
        
    return {'Analysis': ans, 'Reqs': Reqs, 'scripts': scripts, 'return_Format': return_format}



# def generate_pictures(state: State) -> State:
#     # state['data_frames'] is a set of data frames created from the tables
#     # we now also have the python scripts to run on the data frames
#     # hence for each data frame, we do the following:
#     # we run the python scripts on the data frames to generate the pictures and save them in the state

# def analyze(state: State) -> State:
#     # state['data_frames'] is a set of data frames created from the tables
#     # we now also have the set of pictures for each of the topics
#     # hence for each data frame, we do the following:
#     # we run the python scripts on the data frames to generate the analysis and save them in the state


# # def 



graph_builder.add_node(into_data_frames, "into_data_frames")
graph_builder.add_node(generate_analysis, "generate_analysis")

graph_builder.add_edge("into_data_frames", "generate_analysis")

graph_builder.set_entry_point("into_data_frames")
graph_builder.set_finish_point("generate_analysis")

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
        'return_format': {}, # set of dictionaries of data frames and the return format to run on them 
        'Analysis': {},
        'Pictures': {},
        'Pictures_Analysis': {},
        'Reqs': {},
        'DF_Info': {}

    }

    # Run the graph with the sample state
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
        }
    
    print(initial_state)
    final_state2 = graph2.invoke(initial_state)
    
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
        'Pictures_Analysis': final_state2.get('Pictures_Analysis', {})
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