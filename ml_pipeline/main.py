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
from src.data_collection import WebScraper, PDFExtractor
from src.data_processing import DataCleaner, QualityFilter, Normalizer, StorageManager, MetadataHandler, Deduplicator, QAGenerator, QAGenerationConfig
from src.training.dataset_preparation import QADatasetPreparer
from src.training import load_model, LoRATrainer, TrainingLoop, ExperimentTracker, TrainingConfig, WandbCallback
from src.evaluation import BenchmarkCreator, BenchmarkGenerator, MetricsCalculator, PerformanceTester, Comparator, ModelEvaluator
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
        metadata_handler = MetadataHandler()
        deduplicator = Deduplicator(similarity_threshold=0.95, method="levenshtein")
        
        processed_data = cleaner.process(
            web_data, 
            text_column="text",
            remove_boilerplate=True,
            filter_sentences=True,
            min_length=30
        )
        processed_data = quality_filter.filter(processed_data, text_column="text")
        processed_data = normalizer.normalize(processed_data, text_column="text")
        
        # Add metadata and source tracking
        documents = processed_data.to_dict('records')
        documents_with_metadata = metadata_handler.add_metadata(documents)
        
        # Validate metadata
        metadata_stats = metadata_handler.validate_metadata(documents_with_metadata)
        logger.info(f"Metadata validation: {metadata_stats}")
        
        # Deduplicate documents
        original_count = len(documents_with_metadata)
        deduplicated_documents = deduplicator.deduplicate(documents_with_metadata, text_column="text")
        final_count = len(deduplicated_documents)
        
        # Get deduplication statistics
        dedup_stats = deduplicator.get_deduplication_stats(original_count, final_count)
        logger.info(f"Deduplication stats: {dedup_stats}")
        
        # Convert back to DataFrame
        processed_data = pd.DataFrame(deduplicated_documents)
        
        # Generate QA pairs from processed documents
        qa_config = QAGenerationConfig(
            model="gpt-4-turbo",
            temperature=0.3,
            max_qa_per_chunk=2,
            include_source=True,
            include_metadata=True
        )
        qa_generator = QAGenerator(config=qa_config)
        
        # Generate QA pairs
        domain = "electric_vehicles"  # Can be made configurable
        qa_pairs = qa_generator.generate_qa_pairs(deduplicated_documents, domain, text_column="text")
        
        # Validate QA pairs
        qa_validation = qa_generator.validate_qa_pairs(qa_pairs)
        logger.info(f"QA validation: {qa_validation}")
        
        # Get QA statistics
        qa_stats = qa_generator.get_qa_stats(qa_pairs)
        logger.info(f"QA generation stats: {qa_stats}")
        
        # Save QA pairs
        qa_generator.save_qa_pairs(qa_pairs, f"{output_dir}/qa_pairs.jsonl")
        
        # Save processed data
        storage.save_to_parquet(processed_data, f"{output_dir}/processed_data.parquet")
        logger.info(f"Processed {len(processed_data)} documents with metadata")
        
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
        
        # Propagate metadata to QA pairs
        qa_pairs = qa_preparer.load_qa_pairs()
        if qa_pairs:
            source_docs = processed_data.to_dict('records')
            qa_pairs_with_metadata = metadata_handler.propagate_metadata_to_qa(qa_pairs, source_docs)
            qa_pairs_with_attribution = metadata_handler.add_attribution_to_qa(qa_pairs_with_metadata)
            
            # Save QA pairs with metadata
            qa_preparer.save_qa_pairs_with_metadata(qa_pairs_with_attribution)
            logger.info(f"Saved {len(qa_pairs_with_attribution)} QA pairs with metadata and attribution")
        
        # Stage 4: Fine-tuning
        logger.info("=== Stage 4: Fine-tuning ===")
        
        # Check if we have training data
        if not os.path.exists(f"{output_dir}/qa_dataset/train.jsonl"):
            logger.error("Training data not found. Skipping fine-tuning.")
            return
        
        # Initialize experiment tracking
        training_config = TrainingConfig(
            base_model=base_model,
            domain=domain,
            lora_rank=16,
            lora_alpha=32,
            learning_rate=2e-4,
            batch_size=4,
            num_epochs=3
        )
        
        experiment_tracker = ExperimentTracker(
            project="ev-charging-finetune",
            config=training_config,
            tags=["lora", "ev-charging", "qa-generation"]
        )
        
        # Log dataset information
        dataset_stats = {
            "train_samples": len(processed_data),
            "qa_pairs_generated": len(qa_pairs) if 'qa_pairs' in locals() else 0,
            "domain": domain,
            "deduplication_reduction": dedup_stats.get("reduction_percentage", 0)
        }
        experiment_tracker.log_dataset_info(dataset_stats)
        
        # Load base model
        model, tokenizer = load_model(base_model, quantization="4bit")
        
        # Log model information
        model_info = {
            "base_model": base_model,
            "model_size": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "quantization": "4bit"
        }
        experiment_tracker.log_model_info(model_info)
        
        # Apply LoRA
        lora_trainer = LoRATrainer(
            model=model,
            r=training_config.lora_rank,
            alpha=training_config.lora_alpha,
            use_qlora=True
        )
        
        # Load datasets
        from datasets import load_dataset
        train_dataset = load_dataset("json", data_files=f"{output_dir}/qa_dataset/train.jsonl")["train"]
        val_dataset = load_dataset("json", data_files=f"{output_dir}/qa_dataset/validation.jsonl")["train"]
        
        # Training loop with experiment tracking
        training_loop = TrainingLoop(
            model=lora_trainer.get_model(),
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=f"{output_dir}/model_checkpoints",
            experiment_tracker=experiment_tracker
        )
        
        # Run training with WandB callback
        training_results = training_loop.run()
        
        # Log final training summary
        training_summary = {
            "total_training_steps": training_results.get("total_steps", 0),
            "final_loss": training_results.get("final_loss", 0),
            "best_eval_loss": training_results.get("best_eval_loss", 0),
            "training_time": training_results.get("training_time", 0)
        }
        experiment_tracker.log_training_summary(training_summary)
        
        # Stage 5: Evaluation
        logger.info("=== Stage 5: Evaluation ===")
        
        # Generate comprehensive benchmark dataset
        benchmark_generator = BenchmarkGenerator(domain=domain)
        
        # Generate standard benchmark questions
        standard_benchmark = benchmark_generator.generate_benchmark(
            num_questions=30,
            difficulty_distribution={"easy": 0.2, "medium": 0.5, "hard": 0.3}
        )
        
        # Generate adversarial benchmark questions
        adversarial_benchmark = benchmark_generator.create_adversarial_benchmark(num_questions=20)
        
        # Combine benchmarks
        all_benchmark_questions = standard_benchmark + adversarial_benchmark
        
        # Validate benchmark quality
        benchmark_validation = benchmark_generator.validate_benchmark()
        logger.info(f"Benchmark validation: {benchmark_validation}")
        
        # Get benchmark statistics
        benchmark_stats = benchmark_generator.get_benchmark_stats()
        logger.info(f"Benchmark statistics: {benchmark_stats}")
        
        # Save benchmark dataset
        benchmark_generator.save_benchmark(f"{output_dir}/benchmark_dataset.jsonl")
        
        # Log benchmark info to WandB
        experiment_tracker.log_dataset_info({
            "benchmark_questions": len(all_benchmark_questions),
            "standard_questions": len(standard_benchmark),
            "adversarial_questions": len(adversarial_benchmark),
            "benchmark_categories": benchmark_stats.get("categories", []),
            "benchmark_difficulties": benchmark_stats.get("difficulties", [])
        })
        
        # Load fine-tuned model
        from peft import PeftModel
        fine_tuned_model = PeftModel.from_pretrained(model, f"{output_dir}/model_checkpoints")
        
        # Initialize model evaluator
        model_evaluator = ModelEvaluator(device="auto")
        
        # Load benchmark dataset
        benchmark_questions = []
        try:
            with open(f"{output_dir}/benchmark_dataset.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        benchmark_questions.append(json.loads(line))
        except FileNotFoundError:
            logger.warning("Benchmark dataset not found, using sample questions")
            benchmark_questions = [
                {"question": "What is EV charging?", "answer": "Electric vehicle charging"},
                {"question": "How fast can Tesla charge?", "answer": "Up to 250kW with Supercharger V3"}
            ]
        
        # Compare fine-tuned model with baseline
        logger.info("Comparing fine-tuned model with baseline...")
        
        try:
            # Load baseline model (same as base model)
            baseline_model, baseline_tokenizer = model_evaluator.load_model(base_model, is_peft=False)
            
            # Perform comprehensive comparison
            comparison_result = model_evaluator.compare_models(
                fine_tuned_model=fine_tuned_model,
                fine_tuned_tokenizer=tokenizer,
                baseline_model=baseline_model,
                baseline_tokenizer=baseline_tokenizer,
                benchmark=benchmark_questions,
                fine_tuned_name="fine_tuned_ev_model",
                baseline_name="baseline_model"
            )
            
            # Save comparison results
            model_evaluator.save_comparison_results(
                comparison_result, 
                f"{output_dir}/model_comparison_results.json"
            )
            
            # Generate and save comparison report
            comparison_report = model_evaluator.generate_comparison_report(comparison_result)
            with open(f"{output_dir}/comparison_report.md", 'w', encoding='utf-8') as f:
                f.write(comparison_report)
            
            # Log comparison metrics to WandB
            experiment_tracker.log_final_metrics({
                "fine_tuned_rouge1": comparison_result.fine_tuned_metrics.get("rouge1", 0),
                "fine_tuned_rouge2": comparison_result.fine_tuned_metrics.get("rouge2", 0),
                "fine_tuned_bleu": comparison_result.fine_tuned_metrics.get("bleu", 0),
                "fine_tuned_exact_match": comparison_result.fine_tuned_metrics.get("exact_match", 0),
                "baseline_rouge1": comparison_result.baseline_metrics.get("rouge1", 0),
                "baseline_rouge2": comparison_result.baseline_metrics.get("rouge2", 0),
                "baseline_bleu": comparison_result.baseline_metrics.get("bleu", 0),
                "baseline_exact_match": comparison_result.baseline_metrics.get("exact_match", 0),
                "rouge1_improvement": comparison_result.improvements.get("rouge1", 0),
                "rouge2_improvement": comparison_result.improvements.get("rouge2", 0),
                "bleu_improvement": comparison_result.improvements.get("bleu", 0),
                "exact_match_improvement": comparison_result.improvements.get("exact_match", 0),
                "latency_improvement": comparison_result.evaluation_summary.get("latency_improvement", 0),
                "throughput_improvement": comparison_result.evaluation_summary.get("throughput_improvement", 0)
            })
            
            logger.info(f"Model comparison completed. ROUGE-1 improvement: {comparison_result.improvements.get('rouge1', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            # Fallback to basic evaluation
            metrics_calc = MetricsCalculator()
            performance_tester = PerformanceTester(fine_tuned_model, tokenizer)
            
            # Test latency
            latency = performance_tester.test_latency("What is EV charging?", n_runs=5)
            logger.info(f"Average latency: {latency:.4f}s")
            
            # Calculate evaluation metrics
            evaluation_metrics = {
                "latency_avg": latency,
                "model_size_mb": sum(p.numel() for p in fine_tuned_model.parameters()) / 1e6,
                "benchmark_samples": len(benchmark_questions)
            }
            
            # Log evaluation metrics to WandB
            experiment_tracker.log_final_metrics(evaluation_metrics)
        
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