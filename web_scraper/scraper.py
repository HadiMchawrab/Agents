from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import validators
import json
from typing import List
from langdetect import detect
from web_scraper.website import WebsiteContent
from web_scraper.web_driver import setup_webdriver, close_browser
from web_scraper.search import search, get_top_results
from web_scraper.output import save_results_to_json, json_for_ai_api


    
def scrape(query: str, driver: webdriver = None, url: str = "https://www.google.com", save_path: str = None):
    if driver is None:
        driver = setup_webdriver()

    search_results, search_links = search(query, driver, url)
    top_results = get_top_results(search_results, search_links, driver)

    close_browser(driver)

    if save_path is not None:
        json_for_ai_api(top_results, save_path)
    else:
        json_for_ai_api(top_results)

    return top_results