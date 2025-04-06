from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def setup_webdriver():
    start = time.time()
    try:
        options = webdriver.ChromeOptions()
        options.headless = False
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36")
        options.add_argument("--disable-blink-features=AutomationControlled")
        driver = webdriver.Chrome(options=options)
        end = time.time()
        print(f"Webdriver setup Time: {(end - start)} seconds")
        return driver
    except Exception as e:
        print(f"Error setting up webdriver: {str(e)}")
        return None

def close_browser(driver: webdriver):
    driver.quit()
    
    
def wait_for_page(driver: webdriver, wait_time: int = 5):
    try:
        WebDriverWait(driver, wait_time).until(
            EC.presence_of_all_elements_located((By.XPATH, "//*"))
        )
    except Exception as e:
        print(f"Timeout while waiting for page to load: {e}")
        return

