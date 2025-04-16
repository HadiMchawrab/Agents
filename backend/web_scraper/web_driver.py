from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from .tools import get_logger, timer

logger = get_logger("web_driver")

def setup_webdriver():
    with timer("Webdriver Setup"):
        try:
            options = webdriver.ChromeOptions()
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36")
            options.add_argument("--disable-blink-features=AutomationControlled")

            driver = webdriver.Chrome(options=options)
            logger.info("Webdriver setup successfully")
            return driver
        except Exception as e:
            logger.error(f"Error setting up webdriver: {str(e)}")
            return None

def close_browser(driver: webdriver):
    try:
        driver.quit()
        logger.info("Browser closed successfully")
    except Exception as e:
        logger.error(f"Error closing browser: {str(e)}")


def wait_for_page(driver: webdriver, wait_time: int = 5):
    try:
        WebDriverWait(driver, wait_time).until(
            EC.presence_of_all_elements_located((By.XPATH, "//*"))
        )
        logger.info("Page loaded successfully")
    except Exception as e:
        logger.warning(f"Timeout while waiting for page to load: {e}")