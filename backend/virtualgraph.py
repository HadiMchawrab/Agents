from typing import TypedDict, List, Dict, Any
from IPython.display import display
from PIL import Image
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from web_scraper.scraper import scrape, scrape_v2, scrape_remote
import sqlite3
import pandas as pd
import os
import json
import io
import logging
from dotenv import load_dotenv
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from anthropic import OverloadedError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class State(TypedDict):
    tables: str 
    analyzed_topics: list[set]
    csv_files: set
    topic: List[str]
    ScrapedArticles: set
    AnalyzedArticles: set[dict[str, dict[str, str]]]
    Relationship: set
    Explanation: set
    ModelsPerTopic: set
    ML_Models1: set
    

graph_builder = StateGraph(State)





def get_table_columns(state: State, db_name: str = 'temp.db') -> str:
    conn = sqlite3.connect(db_name)
    try:
        table_columns = ''
        for csv_file in state["csv_files"]:
            if not os.path.exists(csv_file):
                continue
            table_name = os.path.splitext(os.path.basename(csv_file))[0]            
            df = pd.read_csv(csv_file)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            columns = pd.read_sql_query(f"PRAGMA table_info({table_name})", conn)['name'].tolist()
            table_columns += f'Table {table_name}:\nHas Columns: {",".join(columns)}\n\n'
    except Exception as e:
        raise
    finally:
        conn.close()    
    return {"tables": table_columns}








@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(OverloadedError)
)
def analyze_tables_node(state: State):
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve the API key from the environment
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    if not CLAUDE_API_KEY:
        raise ValueError("CLAUDE_API_KEY environment variable is not set")

    try:
        model = ChatAnthropic(
            model_name="claude-3-7-sonnet-20250219",
            temperature=0,
            anthropic_api_key=CLAUDE_API_KEY,
            max_retries=3
        )
        
        input_messages = [
            SystemMessage(content="""Given tables and columns names, extract the topic of the database. 
                                    Provide 4 possible topics where machine learning models would be implemented on tabular database similar to the one above to improve the performance of such a company.
                                    Hence the 4 topics you are returning are the fields we want to dive into in our web scraper, hence they should be clear and have the key words in place
                                    Return them in a JSON array of objects with topic names and reasoning"""), 
            HumanMessage(content="""Return the response **only** in this strict JSON format, with no additional text or explanations. DON'T GENERATE ANY TEXT OUTSIDE of the json format("Machine learning models employed in" does not change in all of the topics):
            
                                ```json
                                {
                                    "answer": [
                                        {
                                            "topic": "'Topic 1' and why it is important to the company and how it optimizes the performance of the company",
                                            "ML_Models": "ML Models 1 inferred by Claude from the articles",
                                            "reasoning": "Reasoning 1 or the relationship between the topic and the ML Model and what columns and data types could possibly be used in the ML model"
                                        },
                                        {
                                            "topic": "Machine learning models employed in 'Topic 2'",
                                            "ML_Models": "ML Models 2 inferred by Claude from the articles"
                                            "reasoning": "Reasoning 2 or the relationship between the topic and the ML Model and what columns and data types could possibly be used in the ML model"
                                        }
                                    ]
                                }
                                ```
                           """),
            HumanMessage(content=state["tables"])
        ]
        
        ai_message = model.invoke(input_messages)
        content = ai_message.content

        # Extract JSON content
        if "```json" in content:
            start_index = content.find("```json") + len("```json")
            end_index = content.find("```", start_index)
            content = content[start_index:end_index].strip()
        else:
            content = content.strip()

        try:
            json_response = json.loads(content)
            ans = json_response['answer']
            topics = [topic["topic"] for topic in ans]
            ML_Models1 = [topic["ML_Models"] for topic in ans]
            
            logging.info(f"Topics: {topics}")
            logging.info(f"ML Models: {ML_Models1}")
            
            return {
                "analyzed_topics": ans,
                "topic": topics,
                "ML_Models1": ML_Models1
            }
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {content}")
            raise ValueError("Invalid JSON response from Claude") from e

    except Exception as e:
        logging.error(f"Error in analyze_tables_node: {str(e)}")
        raise







def scrape_node(state: State):
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    model = ChatAnthropic(model_name="claude-3-5-haiku-20241022", temperature=0, anthropic_api_key=CLAUDE_API_KEY)
    articles = {}
    ans = {}
    ModelsPerTopic = {}
    i = 0
    logging.info(f"ML Models: {state['ML_Models1']}")
    for topic in (state["topic"]):

        Disected = state["ML_Models1"][i]
        ML_MOD = state["ML_Models1"]
        logging.info(f"ML_MOD: {ML_MOD}") 
        logging.info(f"Disected: {Disected}")
        scr_into_str = "\n".join(scrape_remote(topic))
        articles[topic] = scr_into_str



        Input_messages= [SystemMessage(content = """You are given 3 web pages in 1 json file, summarize the content in a clear and concise manner.
                                                    Focus on the machine learning models used in the web page not previously mentioned in the previously extracted ML models from the topic and how they affect the performance of the company."""), 
                            HumanMessage(content = """DON'T GENERATE ANY TEXT OUTSIDE of the json format, DO NOT output "json", "Json", or any text before the opening brace '{'. Start the output *directly* with {.

                                    
                                    {
                                        "answer": [
                                            {
                                        "Article_Summary": "Topic 1 Articles Summary Generated by Claude"(we want to see the relationship between the ML model and the topic at hand and possible type of data used in the model. infer the type of data used in the model from the articles + your own knowledge),
                                        "ML_Models": "ML Models 1 inferred by Claude from the articles which where not mentioned in my human message or the previous ML Models"(limit the number of models to 3 separated by commas),
                                        }
                                        ]
                                    }  
                                    """),
                            HumanMessage(content = articles[topic]),
                            HumanMessage(content = f"Previous ML Models:{Disected}")]
        
        logging.info(f"Message Sent to AI")
        ai_message = model.invoke(Input_messages)
        logging.info(ai_message.content)
        content = ai_message.content

        # Strip triple backticks if they exist
        if "```json" in content:
            start_index = content.find("```json") + len("```json")
            end_index = content.find("```", start_index)
            content = content[start_index:end_index].strip()
        elif "```" in content:
            # just in case it starts directly with triple backticks
            content = content.strip("```").strip()

        try:
            json_response = json.loads(content)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON from AI message: {content}")
            raise e

        answer = json_response['answer']
        ans[topic] = answer[0]
        ModelsPerTopic[topic] = ans[topic]["ML_Models"]
        i += 1
    return {"ScrapedArticles": articles, "AnalyzedArticles": ans, "ModelsPerTopic": ModelsPerTopic}


def relevance_node(state: State):
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    model = ChatAnthropic(
        model_name="claude-3-7-sonnet-20250219",
        temperature=0,
        anthropic_api_key=CLAUDE_API_KEY
    )

    Relationships = {}
    Explanations = {}

    for i, topic in enumerate(state["topic"]):
        Input_messages = [
            SystemMessage(content="""You are given the ML models used in the topics and how they would benefit the company, 
                                     You are also given the initial tables and column names of my database,
                                     You are given the interpretation of the tables and column names.
                                     Find the relevance of each of the ML models posed with the tables and columns names of the database.
                                  """),
            HumanMessage(content="""Return the response **only** in this strict JSON format, with no additional text or explanations:
                                    ```json
                                    {
                                        "Relationship": "Choose the columns and tables from the initial tables and columns which are relevant to the ML models for the given topic(Make the columns you choose clear in the output)",
                                        "Explanation": "Explains the relationship between the columns names and how they are going to be used in the ML models given in the modelsWeUse"
                                    }
                                    ```"""),
            HumanMessage(content=f"The initial tables and columns: {state['tables']}"),
            HumanMessage(content=json.dumps(state["analyzed_topics"][i])), 
            HumanMessage(content=f"modelsWeUse: {state['ModelsPerTopic'][topic]}")]  

        ai_response = model.invoke(Input_messages)
        raw_content = ai_response.content.strip()
        logging.info(f"Claude raw response: {raw_content}")

        # Extract JSON from fenced block
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

        Relationships[topic] = [parsed_json.get("Relationship", "No Relationship returned")]
        Explanations[topic] = [parsed_json.get("Explanation", "No Explanation returned")]
    

    return {"Relationship": Relationships, "Explanation": Explanations}



# Add nodes to the graph
graph_builder.add_node("extract_tables", get_table_columns)
graph_builder.add_node("analyze_tables", analyze_tables_node)
graph_builder.add_node("scrape_articles", scrape_node)
graph_builder.add_node("relevance", relevance_node)

# Add edges
graph_builder.add_edge("extract_tables", "analyze_tables")
graph_builder.add_edge("analyze_tables", "scrape_articles")
graph_builder.add_edge("scrape_articles", "relevance")

# Set entry and finish points
graph_builder.set_entry_point("extract_tables")
graph_builder.set_finish_point("relevance")


# Compile the graph
graph = graph_builder.compile()

def run_graph(csv_files: List[str]):   
    csv_files = set(csv_files)
    
    initial_state = {
        "tables": "",
        "analyzed_topics": [],
        "csv_files": csv_files,
        "topic": [],
        "ScrapedArticles": {},
        "AnalyzedArticles": {},
        "ModelsPerTopic": {},
        "Relationship": {},
        "Explanation": {},
        "ML_Models1": {}
    }
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    
    print("\nExtracted Tables:")
    print(final_state["tables"])
    print("\nTable Analysis:")
    print(final_state["analyzed_topics"])
    print("\nTopics:")
    print(final_state["topic"])
    print("\n Analyzed Articles:")
    print(final_state["AnalyzedArticles"])
    print("\n Models per Topic:")
    print(final_state["ModelsPerTopic"])
    print("\n Relationships:")
    print(final_state["Relationship"])
    print("\n Explanations:")
    print(final_state["Explanation"])
    
    return final_state

if __name__ == "__main__":
    run_graph(["csv_files/banking.csv, csv_files/data.csv"])
