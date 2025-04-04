from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from web_scraper.scraper import scrape
import sqlite3
import pandas as pd
import os
import json
from keys import CLAUDE_API_KEY
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class State(TypedDict):
    tables: str 
    analyzed_topics: str
    csv_files: set
    topic: List[str]
    ScrapedArticles: set
    AnalyzedArticles: set
    ModelsPerTopic: set
    Relevance: set

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








def analyze_tables_node(state: State):
    model = ChatAnthropic(model_name="claude-3-5-haiku-20241022", temperature=0, anthropic_api_key=CLAUDE_API_KEY)
    input_messages= [SystemMessage(content = """Given tables and columns names, extract the topic of the database. 
                                                Provide possible topics where  machine learning models would be implemented on tabular database similar to the one above to improve the performance of such a company.
                                                Hence the topics you are returning are the fields we want to dive into in our web scraper, hence they should be clear and have the key words in place
                                                Return them in a JSON array of objects with topic names and reasoning"""), 
                     HumanMessage(content = """Return the response **only** in this strict JSON format, with no additional text or explanations. DON'T GENERATE ANY TEXT OUTSIDE of the json format("Machine learning models employed in" does not change in all of the topics):
                    
                                        ```json
                                        {
                                            "answer": [
                                                {
                                                    "topic": "Machine learning models employed in 'Topic 1'",
                                                    "reasoning": "Reasoning 1"
                                                },
                                                {
                                                    "topic": "Machine learning models employed in 'Topic 2'",
                                                    "reasoning": "Reasoning 2"
                                                }
                                            ]
                                        }
                                                ```
                                   """),
                    HumanMessage(content = state["tables"])]
    
    ai_message = model.invoke(input_messages)
    if not ai_message.content.startswith("```json"):
        start_index = ai_message.content.find("```json") + len("```json")
        end_index = ai_message.content.find("```", start_index)
        ai_message = ai_message.content[start_index:end_index].strip()
    else:
        ai_message = ai_message.content[7:-3].strip()
    json_response = json.loads(ai_message)  
    ans = json_response['answer']
    topics = [topic["topic"] for topic in ans]
    return {"analyzed_topics": ans, "topic": topics}







def scrape_node(state: State):
    model = ChatAnthropic(model_name="claude-3-5-haiku-20241022", temperature=0, anthropic_api_key=CLAUDE_API_KEY)
    articles = {}
    ans = {}
    ModelsPerTopic = {}
    for topic in state["topic"]:
        articles[topic] = scrape(topic)
        Input_messages= [SystemMessage(content = """You are given 3 web pages in 1 json file, summarize the content in a clear and concise manner.
                                                Focus on the machine learning models used in the web page and how they affect the performance of the company."""), 
                            HumanMessage(content = """Return the response **only** in this strict JSON format, with no additional text or explanations. DON'T GENERATE ANY TEXT OUTSIDE of the json format
                                    ```Json 
                                    {
                                        "answer": [
                                            {
                                        "Article_Summary": "Topic 1 Articles Summary Generated by Claude",
                                        "ML_Models": "ML Models 1 inferred by Claude from the articles numbered 1-2-3-4 ..."
                                        }
                                        ]
                                    }
                                    ```
                                    """),
                            HumanMessage(content = state["ScrapedArticles"][topic])]
        logging.info(f"Message Sent to AI")
        ai_message = model.invoke(Input_messages)
        logging.info(ai_message.content)
        if not ai_message.content.startswith("```json"):
            start_index = ai_message.content.find("```json") + len("```json")
            end_index = ai_message.content.find("```", start_index)
            ai_message = ai_message.content[start_index:end_index].strip()
        else:
            ai_message = ai_message.content[7:-3].strip()
        json_response = json.loads(ai_message)
        answer = json_response['answer']
        ans[topic] = answer[0]
        ModelsPerTopic[topic] = ans[topic]["ML_Models"]
    return {"ScrapedArticles": articles, "AnalyzedArticles": ans, "ModelsPerTopic": ModelsPerTopic}







# Add nodes to the graph
graph_builder.add_node("extract_tables", get_table_columns)
graph_builder.add_node("analyze_tables", analyze_tables_node)
graph_builder.add_node("scrape_articles", scrape_node)


# Add edges
graph_builder.add_edge("extract_tables", "analyze_tables")
graph_builder.add_edge("analyze_tables", "scrape_articles")


# Set entry and finish points
graph_builder.set_entry_point("extract_tables")
graph_builder.set_finish_point("scrape_articles")


# Compile the graph
graph = graph_builder.compile()

def test_graph():    
    initial_state = {
        "tables": "",
        "analyzed_topics": "",
        "csv_files": {'csv_test/banking.csv','csv_test/data.csv'},
        "topic": [],
        "ScrapedArticles": {},
        "AnalyzedArticles": {},
        "ModelsPerTopic": {}
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
    

if __name__ == "__main__":
    test_graph()
