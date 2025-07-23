import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_pipeline.src.data_collection.web_scraper import WebScraper

def test_robots_txt():
    urls = [
        "https://www.google.com/",           # Allowed
        "https://www.google.com/search?q=ai" # Disallowed
    ]
    scraper = WebScraper(output_dir="test_scraper_output", rate_limit=0.5)
    scraper.collect(urls)

if __name__ == "__main__":
    test_robots_txt()