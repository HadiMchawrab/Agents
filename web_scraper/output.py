import json
from typing import List
from web_scraper.website import Website
import os
import time

def save_results_to_json_detailed(results: List[Website], filepath: str = "tests_results/detailed_scraping_results.json"):
    try:
        start = time.time()
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        json_data = [result.to_dict_detailed() for result in results]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)

        end = time.time()
        print(f"Saving to Json Time: {(end - start)} seconds")
        print(f"Results saved to {filepath}")
    except Exception as e:
        print(f"Failed to save results: {str(e)}")
        

def save_results_to_json(results: List[Website], filepath: str = "tests_results/scraping_results.json"):
    try:
        start = time.time()
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        json_data = {
            f"content{i + 1}": result.to_dict_content()["content"]
            for i, result in enumerate(results)
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)

        end = time.time()
        print(f"Saving to Json Time: {(end - start)} seconds")
        print(f"Results saved to {filepath}")
    except Exception as e:
        print(f"Failed to write JSON: {e}")