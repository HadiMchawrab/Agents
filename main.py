from web_scraper import scrape

if __name__ == "__main__":
    query = input("Enter a search query: ")
    content = scrape(query)
    print("\n📄 Extracted Content from Top Results:\n")
    for i, website in enumerate(content, start=1):
        print(f"🔹 {i}. {website['title']}\n   🔗 {website['url']}")
        print("   📝 Content Preview:\n", website["content"], "...\n")