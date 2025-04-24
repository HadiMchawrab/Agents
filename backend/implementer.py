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
model = ChatAnthropic(model_name="claude-3-7-sonnet-20250219", temperature=0, anthropic_api_key=CLAUDE_API_KEY)


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
                    else:
                        logging.warning(f"None of the requested columns for {table_name} exist in the CSV.")
                except Exception as e:
                    logging.error(f"Error reading {csv_file}: {str(e)}")
            else:
                logging.warning(f"CSV file {csv_file} does not exist. Skipping this table.")
    return {'data_frames': data_frames, 'adjusted_columns': adjusted_columns}



# def generate_analysis(state: State) -> State:
#     # state['data_frames'] is a set of data frames created from the tables
#     # hence for each data frame, we do the following:
#     # get info using df.info() and assess it in terms of the data types we need for the ML models and the topic in general, using CLAUDE and save into state
#     # generate using CLAUDE, some python scripts to run on the dataframe after the analysis has been done and saved in the state
#     input_messages= [SystemMessage(content = """
                                                
#                                                 """), 
#                      HumanMessage(content = """Return the response **only** in this strict JSON format, with no additional text or explanations. DON'T GENERATE ANY TEXT OUTSIDE of the json format("Machine learning models employed in" does not change in all of the topics):
                    
#                                         ```json
#                                         {
#                                             "answer": [
#                                                 {}
#                                             ]
#                                         }
#                                                 ```
#                                    """),

#                     HumanMessage(content = state["tables"])]
    
#     ai_message = model.invoke(input_messages)
#     if not ai_message.content.startswith("```json"):
#         start_index = ai_message.content.find("```json") + len("```json")
#         end_index = ai_message.content.find("```", start_index)
#         ai_message = ai_message.content[start_index:end_index].strip()
#     else:
#         ai_message = ai_message.content[7:-3].strip()
#     json_response = json.loads(ai_message)  
#     ans = json_response['answer']


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

graph_builder.set_entry_point("into_data_frames")
graph_builder.set_finish_point("into_data_frames")

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
        'scripts': {} # set of dictionaries of data franes and the scripts to run on them
    }

    # Run the graph with the sample state
    final_state2 = graph2.invoke(initial_state)
    print(final_state2)

if __name__ == "__main__":
    test_graph2()