#!/usr/bin/env python3
"""
Download required NLTK data for the pipeline
"""

import nltk
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_nltk_data():
    """Download required NLTK data"""
    logger.info("Downloading required NLTK data...")
    
    required_data = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger'
    ]
    
    for data in required_data:
        try:
            logger.info(f"Downloading {data}...")
            nltk.download(data, quiet=True)
            logger.info(f"✅ Downloaded {data}")
        except Exception as e:
            logger.error(f"❌ Failed to download {data}: {e}")
    
    logger.info("NLTK data download completed!")

if __name__ == "__main__":
    download_nltk_data() 