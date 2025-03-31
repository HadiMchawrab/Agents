from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import validators
import json

def setup_webdriver(): 
    options = webdriver.ChromeOptions()
    options.headless = False
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36")
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(options=options)
    return driver

def close_browser(driver: webdriver):
    driver.quit()

def search(query: str, driver: webdriver=setup_webdriver(), url: str = "https://www.google.com"):
    if not validators.url(url):
        raise ValueError(f"Invalid URL: {url}")
    try:
        driver.get(url)
        search_box = driver.find_element(By.NAME, "q") # this only works for google
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        time.sleep(5)
        
        search_results = driver.find_elements(By.CSS_SELECTOR, "h3")
        search_links = driver.find_elements(By.CSS_SELECTOR, "div.yuRUbf a")
    
        return search_results, search_links
    except Exception as e:
        print(f"Error searching for {query}: {str(e)}")
        return [], []

def get_top_results(search_results, search_links, limit = 3):
    top_results = []
    for i in range(min(len(search_results), len(search_links), limit)):
        title = search_results[i].text
        url = search_links[i].get_attribute("href")
        top_results.append((title, url))
    return top_results

def extract_content(top_results, driver: webdriver=setup_webdriver()):
    content = []
    for i, (title, url) in enumerate(top_results, start=1):
        website = {
            "id": i,
            "title": title,
            "url": url,
            "headings": [],
            "content": []
        }
        
        driver.get(url)
        time.sleep(5)
        
        heading_tags = ["h1", "h2", "h3", "h4", "h5", "h6"]
        headings = []
        for tag in heading_tags:
            elements = driver.find_elements(By.TAG_NAME, tag)
            headings.extend([h.text.strip() for h in elements if h.text.strip()])
        website["headings"] = headings

        # Extract paragraph text
        paragraphs = driver.find_elements(By.TAG_NAME, "p")
        website["content"] = [p.text.strip() for p in paragraphs if p.text.strip()]

        content.append(website)
        
        driver.quit()

        with open("content.json", "w", encoding="utf-8") as json_file:
            json.dump(content, json_file, indent=4, ensure_ascii=False)

        return content