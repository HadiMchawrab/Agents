from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import validators
from typing import List
from .website import Website
from .web_driver import wait_for_page, wait_for_element
from .tools import get_logger, timer
import os

logger = get_logger("search")


def search_remote_duck(query: str, driver: webdriver, url: str = "https://duckduckgo.com/"):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    import time

    try:
        with timer("DuckDuckGo Search"):
            driver.get(url)

            # Wait until the search box is ready
            wait_for_element(driver, "//input[@id='searchbox_input' or @name='q']")

            # Locate the actual input
            try:
                search_box = driver.find_element(By.ID, "searchbox_input")
            except:
                search_box = driver.find_element(By.NAME, "q")

            search_box.clear()
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)

            # Wait for results to appear
            wait_for_element(driver, "//a[@data-testid='result-title-a']")

            search_elements = driver.find_elements(By.XPATH, "//a[@data-testid='result-title-a']")

            search_results = [el.text for el in search_elements]
            search_links = [el.get_attribute("href") for el in search_elements]

            if not search_links:
                logger.warning("No search results found — dumping HTML for inspection.")
                os.makedirs("debug_pages", exist_ok=True)
                with open("debug_pages/last_failed_duck.html", "w", encoding="utf-8") as f:
                    f.write(driver.page_source)

            logger.info(f"DuckDuckGo search completed with {len(search_links)} results.")
            return search_results, search_links


    except Exception as e:
        logger.exception(f"Error during DuckDuckGo search for '{query}': {e}")
        raise e

    


def search(query: str, driver: webdriver, url: str = "https://www.google.com"):
    if not validators.url(url):
        logger.error(f"Invalid URL: {url}")
        raise ValueError(f"Invalid URL: {url}")

    try:
        with timer("Google Search"):
            driver.get(url)
            wait_for_page(driver)

            search_box = driver.find_element(By.NAME, "q")
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)

            wait_for_page(driver)

            search_results = driver.find_elements(By.CSS_SELECTOR, "h3")
            search_results = [result.text for result in search_results]
            search_links = driver.find_elements(By.CSS_SELECTOR, "div.yuRUbf a")
            search_links = [link.get_attribute("href") for link in search_links]

            logger.info(f"Search completed with {len(search_results)} results.")
            return search_results, search_links

    except Exception as e:
        logger.exception(f"Error occurred while searching for '{query}': {e}")
        raise e

def get_top_results(search_results: List, search_links: List, driver: webdriver, lang: str = "en", limit: int = 3):
    from time import sleep
    top_results: List[Website] = []
    index = 0

    with timer("Top Results Extraction"):
        while len(top_results) < limit and index < len(search_results):
            title = search_results[index]
            url = search_links[index]

            website = Website(title=title, url=url)

            try:
                logger.info(f"Extracting content from result #{index + 1}: {url}")
                website.extract_content(driver)
                if website.content:
                    top_results.append(website)
                    logger.debug(f"Content extracted for: {url}")
                else:
                    logger.debug(f"No content found for: {url}")

            except Exception as e:
                logger.warning(f"Skipping {website.url} due to error: {e}")
                
            index += 1
            sleep(0.5)

    logger.info(f"Total extracted results: {len(top_results)}")
    return top_results


def get_top_results_v2(search_results: List, search_links: List, driver: webdriver, lang: str = "en", limit: int = 3):
    from time import sleep
    top_results: List[Website] = []
    index = 0

    with timer("Top Results Extraction"):
        while len(top_results) < limit and index < len(search_results):
            title = search_results[index]
            url = search_links[index]

            website = Website(title=title, url=url)

            try:
                logger.info(f"Extracting content from result #{index + 1}: {url}")
                website.extract_content_v2(driver)
                if website.content:
                    top_results.append(website)
                    logger.debug(f"Content extracted for: {url}")
                else:
                    logger.debug(f"No content found for: {url}")

            except Exception as e:
                logger.warning(f"Skipping {website.url} due to error: {e}")
                
            index += 1
            sleep(0.5)

    logger.info(f"Total extracted results: {len(top_results)}")
    return top_results


