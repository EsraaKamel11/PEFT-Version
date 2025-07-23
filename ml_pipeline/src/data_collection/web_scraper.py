import time
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from .base_collector import BaseCollector
import os

class WebScraper(BaseCollector):
    def __init__(self, output_dir: str, rate_limit: float = 1.0):
        super().__init__(output_dir)
        self.rate_limit = rate_limit  # seconds between requests

    def collect(self, urls: List[str]) -> None:
        for url in urls:
            self.logger.info(f"Scraping URL: {url}")
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)
                metadata = {
                    'source': url,
                    'status_code': response.status_code,
                    'timestamp': time.time()
                }
                filename = self._sanitize_filename(url) + '.json'
                self.save_data(text, metadata, filename)
            except Exception as e:
                self.logger.error(f"Error scraping {url}: {e}")
            time.sleep(self.rate_limit)

    def _sanitize_filename(self, url: str) -> str:
        import re
        return re.sub(r'[^a-zA-Z0-9]', '_', url)

    def resume(self, urls: List[str]) -> None:
        # Example: skip URLs already scraped
        scraped = set()
        for fname in os.listdir(self.output_dir):
            if fname.endswith('.json'):
                scraped.add(fname.replace('.json', ''))
        to_scrape = [url for url in urls if self._sanitize_filename(url) not in scraped]
        self.logger.info(f"Resuming. {len(to_scrape)} URLs left.")
        self.collect(to_scrape) 