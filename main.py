from web_scraper import scrape
import time

if __name__ == "__main__":
    queries = [
        'Machine learning models employed in Bankruptcy Prediction'
    ]
    for query in queries:           
        start = time.time()
        content = scrape(query, save_path=f"{query}.json")
        end = time.time()
        print(f"Time Taken: {(end - start)/60} minutes")
        print(content)