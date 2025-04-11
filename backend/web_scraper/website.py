from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from web_scraper.web_driver import wait_for_page
from langdetect import detect
from typing import List
import time

class Website:
    def __init__(self, title: str, url: str, content: str = None):
        self.title = title
        self.url = url
        self.content = content
        
    # def iframe_handler(self, driver: webdriver):
    #     try:
    #         iframes = driver.find_elements(By.TAG_NAME, "iframe")
    #         for iframe in iframes:
    #             driver.switch_to.frame(iframe)
    #             self.iframe_handler(driver)
    #             driver.switch_to.default_content()
            

    def check_language(self, driver: webdriver, lang: str = "en"):
        start = time.time()
        lang = lang[:2].lower()

        try:
            html_lang = driver.find_element(By.TAG_NAME, "html").get_attribute("lang")
            if html_lang and not html_lang.startswith(lang):
                print(f"Skipping {self.url} due to HTML language: {html_lang}")
                return False
        except Exception as e:
            print(f"Could not detect lang from HTML tag: {e}")
        end = time.time()
        print(f"Language check Time: {(end - start)} seconds")
        return True

    def extract_content(self, driver: webdriver):
        start = time.time()
        driver.get(self.url)
        print("Waiting for page to load")
        wait_for_page(driver)

        if not self.check_language(driver, lang="en"):
            self.content = None
            return

        print("Language check passed!")
                
        elements = driver.find_elements(By.XPATH, "//*")

        content = ""
        current_heading = None
        
        elements = driver.find_elements(By.XPATH, "//*")
 
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

        end = time.time()
        print(f"One Website Extraction Time: {(end - start)} seconds")

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