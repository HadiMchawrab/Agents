from web_scraper import scrape
import time

if __name__ == "__main__":
    query = "AI solutions for cybersecurity threats in developing countries"
    start = time.time()
    content = scrape(query)
    end = time.time()
    print(f"Time Taken: {end - start}")
    print("\nExtracted Content from Top Results:\n")
    for i, website in enumerate(content, start=1):
        print(f"{i}. {website.title}\n  {website.url}")
        print("   Content Preview:\n", website.content[:500], "...\n")