from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from web_scraper.web_driver import wait_for_page
from langdetect import detect
from typing import List, Optional
import time


class WebsiteContent:
    def __init__(self, title: str, url: str, content: Optional[str] = None):
        self.title = title
        self.url = url
        self.content = content
        self.sections = []

    def extract_text_from_frame(self, driver: webdriver):
        content = ""
        current_heading = None
        sections = []

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
                if current_heading:
                    content += f"{text}\n"
                else:
                    content += f"{text}\n"

                sections.append({
                    "heading": current_heading,
                    "paragraph": text
                })

        return content, sections

    def get_all_paragraphs(self, driver: webdriver):
        paragraphs = [p.text.strip() for p in driver.find_elements(By.TAG_NAME, "p") if p.text.strip()]

        # Handle iframes
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        for idx, iframe in enumerate(iframes):
            try:
                driver.switch_to.frame(iframe)
                paragraphs += self.get_all_paragraphs(driver)
                driver.switch_to.default_content()
            except Exception as e:
                print(f"Skipping iframe {idx}: {e}")
                driver.switch_to.default_content()
        return paragraphs

    def check_language(self, driver: webdriver, lang: str = "en", sample_paragraphs: int = 3):
        lang = lang[:2].lower()

        try:
            html_lang = driver.find_element(By.TAG_NAME, "html").get_attribute("lang")
            if html_lang and not html_lang.startswith(lang):
                print(f"Skipping {self.url} due to HTML language: {html_lang}")
                return False
        except Exception as e:
            print(f"Could not detect lang from HTML tag: {e}")

        paragraphs = self.get_all_paragraphs(driver)
        samples = paragraphs[:sample_paragraphs]
        if not samples:
            print(f"No paragraph samples for lang detection on: {self.url}")
            return False

        combined_text = " ".join(samples)
        try:
            detected = detect(combined_text)
            if not detected.startswith(lang):
                print(f"Skipping {self.url} due to detected content language: {detected}")
                return False
        except Exception as e:
            print(f"Language detection failed: {e}")
            return False

        return True

    def extract_content(self, driver: webdriver):
        driver.get(self.url)
        print("Waiting for page to load")
        wait_for_page(driver)

        if not self.check_language(driver, lang="en", sample_paragraphs=3):
            self.content = None
            return

        print("Language check passed!")

        content, sections = self.extract_text_from_frame(driver)

        # Handle iframes
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        for idx, iframe in enumerate(iframes):
            try:
                driver.switch_to.frame(iframe)
                iframe_content, iframe_sections = self.extract_text_from_frame(driver)
                content += "\n" + iframe_content
                sections += iframe_sections
                driver.switch_to.default_content()
            except Exception as e:
                print(f"Skipping iframe {idx} during content extraction: {e}")
                driver.switch_to.default_content()

        self.content = content
        self.sections = sections

    def to_dict(self):
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "sections": self.sections
        }
