from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from .tools import get_logger, timer
<<<<<<< HEAD
=======
import os
import tempfile
import shutil
import uuid
from selenium.webdriver.remote.webdriver import WebDriver as RemoteWebDriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from webdriver_manager.chrome import ChromeDriverManager

>>>>>>> 70e1b2a288c5fa460b8e61263608bc5032ec3565

logger = get_logger("web_driver")

def setup_webdriver():
    with timer("Webdriver Setup"):
        try:
            options = webdriver.ChromeOptions()
<<<<<<< HEAD
            options.add_argument("--headless=new")
=======
            
            if headless:
                options.add_argument("--headless=new")

>>>>>>> 70e1b2a288c5fa460b8e61263608bc5032ec3565
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36")
            options.add_argument("--disable-blink-features=AutomationControlled")

<<<<<<< HEAD
            driver = webdriver.Chrome(options=options)
            logger.info("Webdriver setup successfully")
            return driver
=======
            # Set binary location for Docker
            if os.environ.get('CHROME_BIN'):
                options.binary_location = os.environ['CHROME_BIN']

            logger.info("Attempting to initialize Chrome WebDriver...")
            driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=options)
            # Verify the driver is working
            driver.get("about:blank")
            if driver.title != "":
                raise Exception("Failed to initialize Chrome WebDriver properly")

            logger.info("Webdriver setup successfully")
            return driver, None
        
>>>>>>> 70e1b2a288c5fa460b8e61263608bc5032ec3565
        except Exception as e:
            logger.error(f"Error setting up webdriver: {str(e)}")
            return None

<<<<<<< HEAD
=======
    
>>>>>>> 70e1b2a288c5fa460b8e61263608bc5032ec3565
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