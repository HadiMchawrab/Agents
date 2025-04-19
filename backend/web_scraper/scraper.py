from selenium import webdriver
from .web_driver import setup_webdriver, close_browser
from .search import search, get_top_results, get_top_results_v2
from .output import save_results_to_json
from .tools import get_logger, timer

logger = get_logger("scraper")

def scrape(query: str, driver: webdriver = None, url: str = "https://www.google.com", save_path: str = None, limit: int = 3, headless: bool = False) -> list:
    logger.info(f"Starting scrape for query: '{query}'")
    
    with timer("Full scrape operation"):
        try:
            local_driver = False
            if driver is None:
                try:
                    driver = setup_webdriver(headless)
                    local_driver = True
                except Exception as e:
                    logger.error(f"Failed to setup webdriver: {e}", exc_info=True)
                    raise e

            search_results, search_links = search(query, driver, url)
            logger.info(f"Search returned {len(search_links)} links")

            top_results = get_top_results(search_results, search_links, driver, limit)
            logger.info(f"Top results extracted: {len(top_results)}")

            if local_driver:
                close_browser(driver)

            save_path = save_path or "scraping_results_" + query.replace(" ", "_") + ".json"
            save_results_to_json(top_results, save_path)

            logger.info(f"Scrape completed successfully. Results saved to: {save_path}")
            return [str(website) for website in top_results]

        except Exception as e:
            logger.error(f"Scraping failed: {e}", exc_info=True)
            return []
        
        
def scrape_v2(query: str, driver: webdriver = None, url: str = "https://www.google.com", save_path: str = None, limit: int = 3, headless: bool = False) -> list:
    logger.info(f"Starting scrape for query: '{query}'")
    
    with timer("Full scrape operation"):
        try:
            local_driver = False
            if driver is None:
                try:
                    driver = setup_webdriver(headless)
                    local_driver = True
                except Exception as e:
                    logger.error(f"Failed to setup webdriver: {e}", exc_info=True)
                    raise e

            search_results, search_links = search(query, driver, url)
            logger.info(f"Search returned {len(search_links)} links")

            top_results = get_top_results_v2(search_results, search_links, driver, limit)
            logger.info(f"Top results extracted: {len(top_results)}")

            if local_driver:
                close_browser(driver)

            save_path = save_path or "scraping_results_" + query.replace(" ", "_") + ".json"
            save_results_to_json(top_results, save_path)

            logger.info(f"Scrape completed successfully. Results saved to: {save_path}")
            return [str(website) for website in top_results]

        except Exception as e:
            logger.error(f"Scraping failed: {e}", exc_info=True)
            return []

