import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from .web_driver import wait_for_page, wait_for_element
from langdetect import detect
from typing import List
from .tools import get_logger, timer
import os
from bs4 import BeautifulSoup
import datetime


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
                raise e
            
            return True

    def extract_content(self, driver: webdriver):
        with timer("One Website Extraction Time"):
            try:
                logger.info(f"Loading {self.url}")
                
                try:
                    driver.get(self.url)
                    wait_for_element(driver, "//p | //h1 | //h2 | //h3")
                except Exception:
                    logger.warning("Initial load failed. Retrying once...")
                    driver.refresh()
                    wait_for_element(driver, "//p | //h1 | //h2 | //h3")
                    
                wait_for_page(driver)

                if not self.check_language(driver, lang="en"):
                    self.content = None
                    return

                logger.info("Language check passed!")
                content = ""
                current_heading = None
                
                for element in driver.find_elements(By.XPATH, "//p | //h1 | //h2 | //h3"):
                    try:
                        tag = element.tag_name.lower()
                        text = element.text.strip()

                        if not text:
                            continue

                        if tag in {"h1", "h2", "h3"}:
                            current_heading = text
                            content += f"\n{text}\n"
                        elif tag == "p":
                            current_heading = text
                            content += "\n" + current_heading + "\n"
                        elif tag == "p" and current_heading is not None:
                            content += text + "\n"
                            current_heading = None
                    except Exception as e:
                        logger.warning(f"Stale or inaccessible element skipped: {e}")
                        continue
                self.content = content
                if not self.content:
                    logger.warning(f"No content found for {self.url}. Saving page source.")
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_name = f"{ts}_debug_failed_{self.title[:20].replace(' ', '_')}.html"
                    os.makedirs("debug_pages", exist_ok=True)
                    with open(os.path.join("debug_pages", file_name), "w", encoding="utf-8") as f:
                        f.write(driver.page_source)

            except Exception as e:
                logger.error(f"Error extracting content from {self.url}: {e}", exc_info=True)
                self.content = None
    
    def extract_content_v2(self, driver: webdriver):

        with timer("One Website Extraction Time [V2]"):
            try:
                logger.info(f"[V2] Loading {self.url}")
                
                try:
                    driver.get(self.url)
                    wait_for_element(driver, "//p | //h1 | //h2 | //h3")
                except Exception:
                    logger.warning("[V2] Initial load failed. Retrying once...")
                    driver.refresh()
                    wait_for_element(driver, "//p | //h1 | //h2 | //h3")
                    
                wait_for_page(driver)

                # Check language before parsing
                if not self.check_language(driver, lang="en"):
                    self.content = None
                    return

                # Use BeautifulSoup to parse the rendered page
                html = driver.page_source
                soup = BeautifulSoup(html, "html.parser")

                logger.info("[V2] Parsing page with BeautifulSoup...")
                content = ""
                for tag in soup.find_all(["h1", "h2", "h3", "p"]):
                    text = tag.get_text(strip=True)
                    if text:
                        if tag.name in ["h1", "h2", "h3"]:
                            content += f"\n{text}\n"
                        elif tag.name == "p":
                            content += text + "\n"

                self.content = content.strip()

                if not self.content:
                    logger.warning(f"[V2] No content found for {self.url}. Saving page source and screenshot.")
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_title = self.title[:20].replace(' ', '_')
                    base_name = f"{ts}_debug_failed_{safe_title}"

                    os.makedirs("debug_pages", exist_ok=True)
                    with open(os.path.join("debug_pages", f"{base_name}.html"), "w", encoding="utf-8") as f:
                        f.write(html)
                    driver.save_screenshot(os.path.join("debug_pages", f"{base_name}.png"))

            except Exception as e:
                logger.error(f"[V2] Error extracting content from {self.url}: {e}", exc_info=True)
                self.content = None

            
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