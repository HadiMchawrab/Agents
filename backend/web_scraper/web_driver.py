from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from .tools import get_logger, timer
import os
import tempfile
import shutil
import uuid
from selenium.webdriver.remote.webdriver import WebDriver as RemoteWebDriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

logger = get_logger("web_driver")



def setup_webdriver_remote():
    with timer("Remote Webdriver Setup"):
        try:
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")

            options.set_capability("browserName", "chrome")

            driver = webdriver.Remote(
                command_executor="http://selenium:4444/wd/hub",
                options=options
            )

            logger.info("Remote WebDriver connected")
            return driver
        except Exception as e:
            logger.error(f"Error setting up remote webdriver: {str(e)}", exc_info=True)
            raise e


def setup_webdriver(headless: bool = True):
    with timer("Webdriver Setup"):
        try:
            options = webdriver.ChromeOptions()
            
            user_data_dir = tempfile.mkdtemp(prefix=f"profile_{uuid.uuid4().hex[:8]}_")
            options.add_argument(f"--user-data-dir={user_data_dir}")
            options.add_argument("--no-first-run")
            options.add_argument("--no-default-browser-check")
            
            if headless:
                options.add_argument("--headless=new")

            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-setuid-sandbox")
            options.add_argument("--disable-web-security")
            options.add_argument("--disable-features=VizDisplayCompositor")
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--enable-unsafe-swiftshader")


            # Set binary location for Docker
            if os.environ.get('CHROME_BIN'):
                options.binary_location = os.environ['CHROME_BIN']

            logger.info("Attempting to initialize Chrome WebDriver...")
            driver = webdriver.Chrome(options=options)

            # Verify the driver is working
            driver.get("about:blank")
            if driver.title != "":
                raise Exception("Failed to initialize Chrome WebDriver properly")

            logger.info("Webdriver setup successfully")
            return driver, user_data_dir
        
        except Exception as e:
            logger.error(f"Error setting up webdriver: {str(e)}", exc_info=True)
            raise e

    
def close_browser(driver: webdriver, user_data_dir: str = None):
    try:
        driver.quit()
        logger.info("Browser closed successfully")

        if user_data_dir and os.path.exists(user_data_dir):
            shutil.rmtree(user_data_dir, ignore_errors=True)
            logger.info(f"Cleaned up user data directory: {user_data_dir}")

    except Exception as e:
        logger.error(f"Error closing browser: {str(e)}", exc_info=True)
        raise e

def wait_for_page(driver: webdriver, wait_time: int = 7):
    try:
        WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.XPATH, "//*"))
        )
        logger.info("Page loaded successfully")
    except Exception as e:
        logger.warning(f"Timeout while waiting for Google Search page: {e}")
        raise e
    
def wait_for_element(driver: webdriver, element: str, wait_time: int = 7):
    try:
        WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.XPATH, element))
        )
        logger.info("Element loaded successfully")
    except Exception as e:
        logger.warning(f"Timeout while waiting for element: {e}")
        raise e