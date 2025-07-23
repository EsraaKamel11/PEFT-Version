#!/usr/bin/env python3
"""
End-to-End LLM Fine-tuning Pipeline for EV Charging Stations
"""

import os
import logging
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.settings import settings, logger
from src.data_collection import WebScraper, PDFExtractor, MetadataHandler
from src.data_processing import DataCleaner, QualityFilter, Normalizer, StorageManager
from src.training.dataset_preparation import QADatasetPreparer
from src.training import load_model, LoRATrainer, TrainingLoop, ExperimentTracker
from src.evaluation import BenchmarkCreator, MetricsCalculator, PerformanceTester, Comparator
from src.deployment import ModelRegistry
from src.orchestration import run_pipeline

def create_sample_data():
    """Create sample EV charging data for testing"""
    sample_texts = [
        "Electric vehicle charging stations are essential infrastructure for the transition to sustainable transportation. Level 1 charging uses a standard 120-volt outlet and provides 2-5 miles of range per hour.",
        "Level 2 charging stations use 240-volt power and can provide 10-60 miles of range per hour, making them ideal for home and workplace charging.",
        "DC fast charging, also known as Level 3 charging, can provide 60-80% charge in 20-30 minutes, making it suitable for long-distance travel.",
        "Tesla Superchargers are proprietary DC fast charging stations that can provide up to 200 miles of range in 15 minutes for compatible vehicles.",
        "Public charging networks like ChargePoint, EVgo, and Electrify America provide access to charging stations across the country.",
        "The cost of charging an electric vehicle varies by location and charging speed, typically ranging from $0.10 to $0.30 per kWh.",
        "Most electric vehicles come with a portable Level 1 charger that can be plugged into any standard electrical outlet.",
        "Charging station connectors include Type 1 (J1772), Type 2 (Mennekes), CHAdeMO, and CCS, with different connectors used in different regions.",
        "Smart charging allows vehicles to charge during off-peak hours when electricity rates are lower, helping to reduce charging costs.",
        "Bidirectional charging technology enables electric vehicles to serve as mobile energy storage, providing power back to the grid during peak demand."
    ]
    
    return pd.DataFrame({
        "text": sample_texts,
        "source": ["sample_data"] * len(sample_texts),
        "timestamp": [pd.Timestamp.now()] * len(sample_texts)
    })

def main():
    """Main pipeline execution"""
    logger.info("Starting EV Charging Stations LLM Pipeline")
    
    # Configuration
    domain = "electric vehicle charging stations"
    base_model = "microsoft/DialoGPT-medium"  # Open access model
    output_dir = "pipeline_output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Stage 1: Data Collection
        logger.info("=== Stage 1: Data Collection ===")
        
        # Web scraping for EV charging info (with fallback to sample data)
        ev_urls = [
            "https://httpbin.org/html",  # Safe test URL
            "https://www.electrifyamerica.com/",
            "https://www.chargepoint.com/"
        ]
        
        web_scraper = WebScraper(output_dir=f"{output_dir}/raw_web")
        web_scraper.collect(ev_urls)
        
        # PDF extraction (if you have EV charging PDFs)
        pdf_paths = []  # Add your PDF paths here
        if pdf_paths:
            pdf_extractor = PDFExtractor(output_dir=f"{output_dir}/raw_pdfs")
            pdf_extractor.collect(pdf_paths)
        
        # Stage 2: Data Processing
        logger.info("=== Stage 2: Data Processing ===")
        
        # Try to load scraped data, fallback to sample data if scraping failed
        storage = StorageManager()
        try:
            # Try to load from the web scraper output directory
            import json
            import glob
            
            web_data_files = glob.glob(f"{output_dir}/raw_web/*.json")
            if web_data_files:
                web_data_list = []
                for file_path in web_data_files:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        web_data_list.append({
                            "text": data.get("data", ""),
                            "source": data.get("metadata", {}).get("source", "unknown"),
                            "timestamp": data.get("metadata", {}).get("timestamp", pd.Timestamp.now())
                        })
                web_data = pd.DataFrame(web_data_list)
                logger.info(f"Loaded {len(web_data)} scraped documents")
            else:
                raise FileNotFoundError("No scraped data found")
                
        except Exception as e:
            logger.warning(f"Could not load scraped data: {e}")
            logger.info("Using sample data for demonstration")
            web_data = create_sample_data()
        
        # Clean and filter
        cleaner = DataCleaner()
        quality_filter = QualityFilter(min_length=50)
        normalizer = Normalizer(model_name=base_model)
        
        processed_data = cleaner.process(
            web_data, 
            text_column="text",
            remove_boilerplate=True,
            filter_sentences=True,
            min_length=30
        )
        processed_data = quality_filter.filter(processed_data, text_column="text")
        processed_data = normalizer.normalize(processed_data, text_column="text")
        
        # Save processed data
        storage.save_to_parquet(processed_data, f"{output_dir}/processed_data.parquet")
        logger.info(f"Processed {len(processed_data)} documents")
        
        # Stage 3: Training Dataset Preparation
        logger.info("=== Stage 3: Training Dataset Preparation ===")
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY not found in environment")
            logger.info("Skipping QA generation - please set OPENAI_API_KEY to continue")
            return
        
        qa_preparer = QADatasetPreparer(
            openai_api_key=openai_api_key,
            domain=domain,
            output_dir=f"{output_dir}/qa_dataset"
        )
        
        texts = processed_data["text"].tolist()
        qa_preparer.prepare(texts, n_questions=3, alpaca_format=True)
        
        # Stage 4: Fine-tuning
        logger.info("=== Stage 4: Fine-tuning ===")
        
        # Check if we have training data
        if not os.path.exists(f"{output_dir}/qa_dataset/train.jsonl"):
            logger.error("Training data not found. Skipping fine-tuning.")
            return
        
        # Load base model
        model, tokenizer = load_model(base_model, quantization="4bit")
        
        # Apply LoRA
        lora_trainer = LoRATrainer(
            model=model,
            r=16,
            alpha=32,
            use_qlora=True
        )
        
        # Load datasets
        from datasets import load_dataset
        train_dataset = load_dataset("json", data_files=f"{output_dir}/qa_dataset/train.jsonl")["train"]
        val_dataset = load_dataset("json", data_files=f"{output_dir}/qa_dataset/validation.jsonl")["train"]
        
        # Training loop
        training_loop = TrainingLoop(
            model=lora_trainer.get_model(),
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=f"{output_dir}/model_checkpoints"
        )
        
        training_loop.run()
        
        # Stage 5: Evaluation
        logger.info("=== Stage 5: Evaluation ===")
        
        # Create benchmark
        benchmark_creator = BenchmarkCreator(domain=domain)
        benchmark_data = benchmark_creator.create_benchmark(n=20)
        
        # Load fine-tuned model
        from peft import PeftModel
        fine_tuned_model = PeftModel.from_pretrained(model, f"{output_dir}/model_checkpoints")
        
        # Evaluate
        metrics_calc = MetricsCalculator()
        performance_tester = PerformanceTester(fine_tuned_model, tokenizer)
        
        # Test latency
        latency = performance_tester.test_latency("What is EV charging?", n_runs=5)
        logger.info(f"Average latency: {latency:.4f}s")
        
        # Stage 6: Deployment
        logger.info("=== Stage 6: Deployment ===")
        
        # Register model
        registry = ModelRegistry()
        registry.register(
            base_model=base_model,
            adapter_path=f"{output_dir}/model_checkpoints",
            version="v1.0",
            metadata={"domain": domain, "performance": {"latency": latency}}
        )
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Model registered: {base_model} v1.0")
        logger.info(f"Output directory: {output_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 