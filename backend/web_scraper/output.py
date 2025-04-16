import json
from typing import List
from backend.web_scraper.website import Website
import os
import time
from backend.web_scraper.tools import get_logger, timer

logger = get_logger("save_json")

def save_results_to_json_detailed(results: List[Website], filepath: str = "tests_results/detailed_scraping_results.json"):
    try:
        with timer("Save Detailed JSON"):
            directory = os.path.dirname(filepath)
            if directory:
                os.makedirs(directory, exist_ok=True)

            json_data = [result.to_dict_detailed() for result in results]
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)

            logger.info(f"Detailed results saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save detailed results: {e}", exc_info=True)


def save_results_to_json(results: List[Website], filepath: str = "tests_results/scraping_results.json"):
    try:
        with timer("Save Simple JSON"):
            directory = os.path.dirname(filepath)
            if directory:
                os.makedirs(directory, exist_ok=True)

            json_data = {
                f"content{i + 1}": result.to_dict_content()["content"]
                for i, result in enumerate(results)
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)

            logger.info(f"Results saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to write JSON: {e}", exc_info=True)