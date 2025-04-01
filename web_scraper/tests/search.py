from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Setup WebDriver options
options = webdriver.ChromeOptions()
options.headless = False  # Run with a visible browser to solve CAPTCHA if needed
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36")  # Fake user-agent
options.add_argument("--disable-blink-features=AutomationControlled")  # Prevent detection

# Initialize WebDriver
driver = webdriver.Chrome(options=options)

# Open Google
driver.get("https://www.google.com")

# Find the search box and enter a query
search_box = driver.find_element(By.NAME, "q")
search_box.send_keys("AI solutions in banking databases")
search_box.send_keys(Keys.RETURN)  # Press Enter

# Wait for the page to load
time.sleep(3)

# Allow manual CAPTCHA solving if detected
input("Solve the CAPTCHA manually (if it appears), then press Enter to continue...")

# Extract search results
search_results = driver.find_elements(By.CSS_SELECTOR, "h3")

# Print search results
print("\n🔍 Top Google Search Results:")
for i, result in enumerate(search_results[:10], start=1):  # Get top 10 results
    print(f"{i}. {result.text}")
