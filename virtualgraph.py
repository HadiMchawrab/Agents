from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
import sqlite3
import pandas as pd
import os
import json
from keys import CLAUDE_API_KEY
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the state type
class State(TypedDict):
    tables: str = ''    
    analyzed_topics: str = ''
    csv_files: set = None
    

# Initialize the graph builder
graph_builder = StateGraph(State)

def get_table_columns(state: State, db_name: str = 'temp.db') -> str:
    if state["csv_files"] is None:
        state["csv_files"] = {'csv_test/banking.csv','csv_test/data.csv'}
    
    # show me "csv_files"
    logging.info(f"csv_files: {state['csv_files']}")
    conn = sqlite3.connect(db_name)
    
    try:
        table_columns = ''
        for csv_file in state["csv_files"]:
            if not os.path.exists(csv_file):
                logging.warning(f"CSV file not found: {csv_file}, skipping")
                continue
            
            logging.info(f"Processing CSV file: {csv_file}")
            table_name = os.path.splitext(os.path.basename(csv_file))[0]
            logging.info(f"Table name: {table_name}")
            
            df = pd.read_csv(csv_file)
            logging.info(f"Read CSV file with {len(df)} rows and {len(df.columns)} columns")
            
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logging.info(f"Imported data to SQL table: {table_name}")
            
            columns = pd.read_sql_query(f"PRAGMA table_info({table_name})", conn)['name'].tolist()
            logging.info(f"Retrieved columns for table {table_name}: {columns}")
            
            table_columns += f'Table {table_name}:\nHas Columns: {",".join(columns)}\n\n'
        
        logging.info("Successfully processed all CSV files")
    except Exception as e:
        logging.error(f"Error in get_table_columns: {str(e)}", exc_info=True)
        raise
    finally:
        conn.close()
        logging.info("Database connection closed")
    
    return {"tables": table_columns}




def analyze_tables_node(state: State):
    model = ChatAnthropic(model_name="claude-3-5-haiku-20241022", temperature=0, anthropic_api_key=CLAUDE_API_KEY)
    input_messages= [SystemMessage(content = """Given tables and columns names, extract the topic of the database. 
                                                Provide possible topics tha include machine learning models that would be implemented on a tabular database similar to the one above to improve the performance of such a company.
                                                Return them in a JSON array of objects with topic names and reasoning."""), 
                     HumanMessage(content = """You need to return the answer in this format 
                                  ```Json 
                                  {
                                    "answer": [
                                        {
                                            "topic": "Topic 1",
                                            "reasoning": "Reasoning 1"
                                        },
                                        {
                                            "topic": "Topic 2",
                                            "reasoning": "Reasoning 2"
                                        }
                                    ]
                                  }
                                  ```
                                   """),
                    HumanMessage(content = state["tables"])]
    
    ai_message = model.invoke(input_messages)
    # ai_message = ai_message.content[7:-3]
    # json_response = json.loads(ai_message)  
    # ans = json_response['answer']
    # topics = [topic["topic"] for topic in ans]
    return {"analyzed_topics": ai_message}


# Add nodes to the graph
graph_builder.add_node("extract_tables", get_table_columns)
graph_builder.add_node("analyze_tables", analyze_tables_node)




# Add edges
graph_builder.add_edge("extract_tables", "analyze_tables")

# Set entry and finish points
graph_builder.set_entry_point("extract_tables")
graph_builder.set_finish_point("analyze_tables")


# Compile the graph
graph = graph_builder.compile()

def test_graph():
    
    # Initialize the state
    initial_state = {
        "tables": "",
        "analyzed_topics": "",
        "csv_files": {'csv_test/banking.csv','csv_test/data.csv'}
    }
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    
    # Print results
    print("\nExtracted Tables:")
    print(final_state["tables"])
    print("\nTable Analysis:")
    print(final_state["analyzed_topics"])

if __name__ == "__main__":
    test_graph()
