import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from .web_driver import wait_for_page
from langdetect import detect
from typing import List
from .tools import get_logger, timer

logger = get_logger(__name__)

class Website:
    def __init__(self, title: str, url: str, content: str = None):
        self.title = title
        self.url = url
        self.content = content     
    
    def check_language(self, driver: webdriver, lang: str = "en"):
        with timer("Language check"):
            lang = lang[:2].lower()

            try:
                html_lang = driver.find_element(By.TAG_NAME, "html").get_attribute("lang")
                if html_lang and not html_lang.startswith(lang):
                    logger.info(f"Skipping {self.url} due to HTML language: {html_lang}")
                    return False
            except Exception as e:
                logger.error(f"Could not detect lang from HTML tag: {e}")
            
            return True

    def extract_content(self, driver: webdriver):
        with timer("One Website Extraction Time"):
            driver.get(self.url)
            logger.info("Waiting for page to load")
            wait_for_page(driver)

            if not self.check_language(driver, lang="en"):
                self.content = None
                return

            logger.info("Language check passed!")
                
            elements = driver.find_elements(By.XPATH, "//*")

            content = ""
            current_heading = None

            for element in elements:
                tag = element.tag_name.lower()
                text = element.text.strip()

                if not text:
                    continue

                if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                    current_heading = text
                    content += f"\n{text}\n"
                elif tag == "p":
                    current_heading = text
                    content += "\n" + current_heading + "\n"
                elif tag == "p" and current_heading is not None:
                    content += text + "\n"
                    current_heading = None
            self.content = content

    def to_dict_detailed(self):
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content
        }

    def to_dict_content(self):
        return {
            "content": self.content
        }

    def __str__(self):
        return str(self.title) + " : " + str(self.content)