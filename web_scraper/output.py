import json
from typing import List
from web_scraper.website import WebsiteContent
import os


def save_results_to_json(results: List[WebsiteContent], filepath: str = "tests_results/scraping_results.json"):
    try:
        
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        json_data = [result.to_dict() for result in results]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {filepath}")
    except Exception as e:
        print(f"Failed to save results: {str(e)}")
