#!/usr/bin/env python3
"""
Prefect Scheduler for LLM Pipeline
Handles automated execution, monitoring, and error recovery
"""

from prefect import flow, task, get_run_logger
from prefect.server.schemas.schedules import CronSchedule
from prefect.deployments import Deployment
from prefect.filesystems import LocalFileSystem
from prefect.blocks.system import Secret
from typing import Dict, Any, Optional
import asyncio
import time
import json
import os
from datetime import datetime, timedelta

# Import pipeline components
from ..data_processing.deduplication import Deduplicator
from ..data_processing.qa_generation import QAGenerator
from ..training.experiment_tracker import ExperimentTracker
from ..evaluation.benchmark_generation import BenchmarkGenerator
from ..evaluation.model_comparison import ModelEvaluator

class PipelineScheduler:
    """Scheduler for LLM pipeline with monitoring and error handling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_run_logger()
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "success": False,
            "errors": [],
            "stage_results": {}
        }
    
    @task(name="initialize-pipeline")
    def initialize_pipeline(self) -> Dict[str, Any]:
        """Initialize pipeline with configuration and monitoring"""
        self.logger.info("🚀 Initializing LLM Pipeline")
        
        # Set up monitoring
        self.metrics["start_time"] = datetime.now()
        
        # Validate configuration
        required_keys = ["data_path", "output_path", "model_name", "openai_api_key"]
        missing_keys = [key for key in required_keys if key not in self.config]
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        # Create output directories
        os.makedirs(self.config["output_path"], exist_ok=True)
        os.makedirs(f"{self.config['output_path']}/models", exist_ok=True)
        os.makedirs(f"{self.config['output_path']}/benchmarks", exist_ok=True)
        os.makedirs(f"{self.config['output_path']}/reports", exist_ok=True)
        
        self.logger.info("✅ Pipeline initialized successfully")
        return {"status": "initialized", "timestamp": datetime.now().isoformat()}
    
    @task(name="load-and-deduplicate-data")
    def load_and_deduplicate_data(self, init_result: Dict[str, Any]) -> Dict[str, Any]:
        """Load and deduplicate input data"""
        self.logger.info("📊 Loading and deduplicating data...")
        
        try:
            # Initialize deduplicator
            deduplicator = Deduplicator(
                similarity_threshold=self.config.get("similarity_threshold", 0.8),
                method=self.config.get("deduplication_method", "hybrid")
            )
            
            # Load data
            with open(self.config["data_path"], 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Deduplicate
            deduplicated_data = deduplicator.deduplicate(data)
            
            # Save deduplicated data
            output_file = f"{self.config['output_path']}/deduplicated_data.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(deduplicated_data, f, indent=2, ensure_ascii=False)
            
            result = {
                "original_count": len(data),
                "deduplicated_count": len(deduplicated_data),
                "reduction_percentage": ((len(data) - len(deduplicated_data)) / len(data)) * 100,
                "output_file": output_file
            }
            
            self.metrics["stage_results"]["deduplication"] = result
            self.logger.info(f"✅ Deduplication completed: {result['reduction_percentage']:.1f}% reduction")
            
            return result
            
        except Exception as e:
            error_msg = f"Deduplication failed: {str(e)}"
            self.logger.error(error_msg)
            self.metrics["errors"].append(error_msg)
            raise
    
    @task(name="generate-qa-pairs")
    def generate_qa_pairs(self, dedup_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate QA pairs from deduplicated data"""
        self.logger.info("🤖 Generating QA pairs...")
        
        try:
            # Load deduplicated data
            with open(dedup_result["output_file"], 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Initialize QA generator
            qa_generator = QAGenerator(
                openai_api_key=self.config["openai_api_key"],
                model=self.config.get("openai_model", "gpt-4"),
                max_retries=self.config.get("max_retries", 3)
            )
            
            # Generate QA pairs
            qa_pairs = qa_generator.generate_qa_pairs(
                data,
                num_questions_per_doc=self.config.get("questions_per_doc", 3),
                include_metadata=self.config.get("include_metadata", True)
            )
            
            # Save QA pairs
            output_file = f"{self.config['output_path']}/qa_pairs.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
            
            result = {
                "qa_pairs_count": len(qa_pairs),
                "output_file": output_file,
                "metadata_included": self.config.get("include_metadata", True)
            }
            
            self.metrics["stage_results"]["qa_generation"] = result
            self.logger.info(f"✅ QA generation completed: {len(qa_pairs)} pairs created")
            
            return result
            
        except Exception as e:
            error_msg = f"QA generation failed: {str(e)}"
            self.logger.error(error_msg)
            self.metrics["errors"].append(error_msg)
            raise
    
    @task(name="setup-experiment-tracking")
    def setup_experiment_tracking(self, qa_result: Dict[str, Any]) -> Dict[str, Any]:
        """Set up experiment tracking with WandB"""
        self.logger.info("📈 Setting up experiment tracking...")
        
        try:
            # Initialize experiment tracker
            experiment_tracker = ExperimentTracker(
                project_name=self.config.get("wandb_project", "llm-pipeline"),
                entity=self.config.get("wandb_entity", None),
                run_name=f"pipeline-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            
            # Log configuration
            experiment_tracker.log_config(self.config)
            
            # Log QA generation results
            experiment_tracker.log_metrics({
                "qa_pairs_generated": qa_result["qa_pairs_count"],
                "metadata_included": qa_result["metadata_included"]
            })
            
            result = {
                "experiment_id": experiment_tracker.run.id,
                "project_name": experiment_tracker.project_name,
                "run_name": experiment_tracker.run_name
            }
            
            self.metrics["stage_results"]["experiment_tracking"] = result
            self.logger.info(f"✅ Experiment tracking setup: {result['experiment_id']}")
            
            return result
            
        except Exception as e:
            error_msg = f"Experiment tracking setup failed: {str(e)}"
            self.logger.error(error_msg)
            self.metrics["errors"].append(error_msg)
            raise
    
    @task(name="generate-benchmark")
    def generate_benchmark(self, tracking_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate benchmark dataset"""
        self.logger.info("🎯 Generating benchmark dataset...")
        
        try:
            # Load QA pairs
            with open(self.config["output_path"] + "/qa_pairs.json", 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
            
            # Initialize benchmark generator
            benchmark_generator = BenchmarkGenerator(
                domain=self.config.get("domain", "general"),
                difficulty_levels=self.config.get("difficulty_levels", ["easy", "medium", "hard"]),
                categories=self.config.get("categories", ["general", "technical", "business"])
            )
            
            # Generate benchmark
            benchmark = benchmark_generator.generate_benchmark(
                qa_pairs,
                num_questions=self.config.get("benchmark_size", 100),
                include_adversarial=self.config.get("include_adversarial", True)
            )
            
            # Save benchmark
            output_file = f"{self.config['output_path']}/benchmarks/benchmark.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(benchmark, f, indent=2, ensure_ascii=False)
            
            result = {
                "benchmark_size": len(benchmark["questions"]),
                "difficulty_distribution": benchmark["statistics"]["difficulty_distribution"],
                "category_distribution": benchmark["statistics"]["category_distribution"],
                "output_file": output_file
            }
            
            self.metrics["stage_results"]["benchmark_generation"] = result
            self.logger.info(f"✅ Benchmark generated: {len(benchmark['questions'])} questions")
            
            return result
            
        except Exception as e:
            error_msg = f"Benchmark generation failed: {str(e)}"
            self.logger.error(error_msg)
            self.metrics["errors"].append(error_msg)
            raise
    
    @task(name="evaluate-models")
    def evaluate_models(self, benchmark_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate models using benchmark"""
        self.logger.info("📊 Evaluating models...")
        
        try:
            # Load benchmark
            with open(benchmark_result["output_file"], 'r', encoding='utf-8') as f:
                benchmark = json.load(f)
            
            # Initialize model evaluator
            model_evaluator = ModelEvaluator(
                metrics=self.config.get("evaluation_metrics", ["rouge", "bleu", "exact_match"]),
                domain_metrics=self.config.get("domain_metrics", True)
            )
            
            # Mock model evaluation (replace with actual model loading)
            evaluation_results = model_evaluator.evaluate_models(
                benchmark,
                models={
                    "baseline": "baseline_model",
                    "fine_tuned": "fine_tuned_model"
                }
            )
            
            # Save evaluation results
            output_file = f"{self.config['output_path']}/reports/evaluation_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
            
            # Generate comparison report
            report_file = f"{self.config['output_path']}/reports/comparison_report.md"
            model_evaluator.generate_comparison_report(evaluation_results, report_file)
            
            result = {
                "evaluation_results_file": output_file,
                "comparison_report_file": report_file,
                "models_evaluated": list(evaluation_results["models"].keys()),
                "overall_improvement": evaluation_results.get("overall_improvement", 0)
            }
            
            self.metrics["stage_results"]["model_evaluation"] = result
            self.logger.info(f"✅ Model evaluation completed: {result['overall_improvement']:.2f}% improvement")
            
            return result
            
        except Exception as e:
            error_msg = f"Model evaluation failed: {str(e)}"
            self.logger.error(error_msg)
            self.metrics["errors"].append(error_msg)
            raise
    
    @task(name="finalize-pipeline")
    def finalize_pipeline(self, eval_result: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize pipeline and generate summary"""
        self.logger.info("🏁 Finalizing pipeline...")
        
        try:
            # Calculate final metrics
            self.metrics["end_time"] = datetime.now()
            duration = self.metrics["end_time"] - self.metrics["start_time"]
            self.metrics["success"] = len(self.metrics["errors"]) == 0
            
            # Generate summary report
            summary = {
                "pipeline_status": "completed" if self.metrics["success"] else "failed",
                "duration_seconds": duration.total_seconds(),
                "stages_completed": len(self.metrics["stage_results"]),
                "errors_count": len(self.metrics["errors"]),
                "stage_results": self.metrics["stage_results"],
                "errors": self.metrics["errors"],
                "timestamp": datetime.now().isoformat()
            }
            
            # Save summary
            summary_file = f"{self.config['output_path']}/pipeline_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Pipeline completed in {duration.total_seconds():.1f} seconds")
            self.logger.info(f"📊 Summary saved to: {summary_file}")
            
            return summary
            
        except Exception as e:
            error_msg = f"Pipeline finalization failed: {str(e)}"
            self.logger.error(error_msg)
            self.metrics["errors"].append(error_msg)
            raise

@flow(name="LLM Pipeline", description="End-to-end LLM fine-tuning pipeline with monitoring")
def main_pipeline_flow(config: Dict[str, Any]) -> Dict[str, Any]:
    """Main pipeline flow with all stages"""
    
    # Initialize scheduler
    scheduler = PipelineScheduler(config)
    
    # Execute pipeline stages
    init_result = scheduler.initialize_pipeline()
    dedup_result = scheduler.load_and_deduplicate_data(init_result)
    qa_result = scheduler.generate_qa_pairs(dedup_result)
    tracking_result = scheduler.setup_experiment_tracking(qa_result)
    benchmark_result = scheduler.generate_benchmark(tracking_result)
    eval_result = scheduler.evaluate_models(benchmark_result)
    final_result = scheduler.finalize_pipeline(eval_result)
    
    return final_result

def create_deployment(
    name: str = "llm-pipeline-deployment",
    schedule: Optional[CronSchedule] = None,
    work_pool_name: str = "default-agent-pool",
    config: Optional[Dict[str, Any]] = None
) -> Deployment:
    """Create a Prefect deployment for the pipeline"""
    
    # Default configuration
    default_config = {
        "data_path": "data/input_data.json",
        "output_path": "outputs",
        "model_name": "microsoft/DialoGPT-medium",
        "openai_api_key": "your-openai-api-key",
        "similarity_threshold": 0.8,
        "deduplication_method": "hybrid",
        "questions_per_doc": 3,
        "include_metadata": True,
        "wandb_project": "llm-pipeline",
        "benchmark_size": 100,
        "include_adversarial": True,
        "evaluation_metrics": ["rouge", "bleu", "exact_match"],
        "domain_metrics": True
    }
    
    if config:
        default_config.update(config)
    
    # Create deployment
    deployment = Deployment.build_from_flow(
        flow=main_pipeline_flow,
        name=name,
        parameters={"config": default_config},
        schedule=schedule,
        work_pool_name=work_pool_name,
        description="Automated LLM pipeline with monitoring and error handling"
    )
    
    return deployment

if __name__ == "__main__":
    # Example deployment with daily schedule
    deployment = create_deployment(
        name="prod-llm-pipeline",
        schedule=CronSchedule(cron="0 3 * * *"),  # Daily at 3AM
        work_pool_name="default-agent-pool",
        config={
            "data_path": "data/ev_charging_data.json",
            "output_path": "outputs/ev_pipeline",
            "domain": "electric_vehicles",
            "wandb_project": "ev-charging-pipeline"
        }
    )
    
    deployment.apply()
    print("✅ Deployment created successfully!") 