from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Setup WebDriver options
options = webdriver.ChromeOptions()
options.headless = False  # Run with a visible browser
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36")
options.add_argument("--disable-blink-features=AutomationControlled")

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

# Extract search results (titles and URLs)
search_results = driver.find_elements(By.CSS_SELECTOR, "h3")
search_links = driver.find_elements(By.CSS_SELECTOR, "div.yuRUbf a")

# Store and display search results
top_results = []
for i in range(min(len(search_results), len(search_links), 3)):  # Limit to 3 results
    title = search_results[i].text
    url = search_links[i].get_attribute("href")
    top_results.append((title, url))

# Visit each search result and extract content
print("\n📄 Extracted Content from Top Results:\n")
for i, (title, url) in enumerate(top_results, start=1):
    print(f"🔹 {i}. {title}\n   🔗 {url}")

    # Open the link
    driver.get(url)
    time.sleep(3)  # Wait for the page to load

    # Extract main content (Paragraphs)
    paragraphs = driver.find_elements(By.TAG_NAME, "p")
    content = "\n".join([p.text for p in paragraphs if p.text.strip()])  # Filter empty text

    # Print first 500 characters of content
    print("   📝 Content Preview:\n", content, "...\n")

# Close the browser
driver.quit()
