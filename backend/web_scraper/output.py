import json
from typing import List
from .website import Website
import os
import time
from .tools import get_logger, timer

logger = get_logger("save_json")

def save_results_to_json_detailed(results: List[Website], filepath: str = "detailed_scraping_results.json"):
    try:
        with timer("Save Detailed JSON"):
            # Construct the path relative to the current directory
            output_dir = "scraping_results"
            os.makedirs(output_dir, exist_ok=True)
            
            # Join the output directory with the filename
            full_path = os.path.join(output_dir, filepath)

            json_data = [result.to_dict_detailed() for result in results]
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)

            logger.info(f"Detailed results saved to {full_path}")
    except Exception as e:
        logger.error(f"Failed to save detailed results: {e}", exc_info=True)


def save_results_to_json(results: List[Website], filepath: str = "scraping_results.json"):
    try:
        with timer("Save Simple JSON"):
            # Construct the path relative to the current directory
            output_dir = "scraping_results"
            os.makedirs(output_dir, exist_ok=True)
            
            # Join the output directory with the filename
            full_path = os.path.join(output_dir, filepath)

            json_data = {
                f"content{i + 1}": result.to_dict_content()["content"]
                for i, result in enumerate(results)
            }

            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)

            logger.info(f"Results saved to {full_path}")
    except Exception as e:
        logger.error(f"Failed to write JSON: {e}", exc_info=True)