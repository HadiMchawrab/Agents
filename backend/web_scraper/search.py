from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import validators
from typing import List
from web_scraper.website import Website
from web_scraper.web_driver import wait_for_page
import time


def search(query: str, driver: webdriver, url: str = "https://www.google.com"):
    start = time.time()
    if not validators.url(url):
        raise ValueError(f"Invalid URL: {url}")

    try:
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
        
        print(search_results, search_links)

        end = time.time()
        print(f"Search Time: {(end - start)} seconds")
        return search_results, search_links

    except Exception as e:
        print(f"Error searching for {query}")
        return [], []


def get_top_results(search_results: List, search_links: List, driver: webdriver, lang: str = "en", limit: int = 3):
    start = time.time()
    top_results: str = []
    index = 0

    while len(top_results) < limit and index < len(search_results):
        title = search_results[index]
        url = search_links[index]

        website = Website(title=title, url=url)

        try:
            print(f"Extracting content from {index}")
            website.extract_content(driver)
            if website.content is not None:
                top_results.append(website)

        except Exception as e:
            print(f"Skipping {website.url} due to error: {e}")

        index += 1

    end = time.time()
    print(f"Extraction Time: {(end - start)} seconds")
    return top_results
