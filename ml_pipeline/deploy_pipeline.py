#!/usr/bin/env python3
"""
Deploy LLM Pipeline with Prefect
Handles deployment configuration, scheduling, and monitoring setup
"""

import os
import json
from datetime import datetime
from prefect.server.schemas.schedules import CronSchedule
from src.deployment.scheduler import create_deployment, main_pipeline_flow

def create_ev_pipeline_deployment():
    """Create EV charging pipeline deployment"""
    
    # EV-specific configuration
    ev_config = {
        "data_path": "data/ev_charging_data.json",
        "output_path": "outputs/ev_pipeline",
        "model_name": "microsoft/DialoGPT-medium",
        "openai_api_key": os.getenv("OPENAI_API_KEY", "your-openai-api-key"),
        "domain": "electric_vehicles",
        "wandb_project": "ev-charging-pipeline",
        "wandb_entity": os.getenv("WANDB_ENTITY", None),
        
        # Data processing
        "similarity_threshold": 0.85,
        "deduplication_method": "hybrid",
        "questions_per_doc": 5,
        "include_metadata": True,
        
        # Benchmark generation
        "benchmark_size": 150,
        "include_adversarial": True,
        "difficulty_levels": ["easy", "medium", "hard"],
        "categories": ["pricing", "technical", "compatibility", "environmental"],
        
        # Evaluation
        "evaluation_metrics": ["rouge", "bleu", "exact_match", "bertscore"],
        "domain_metrics": True,
        
        # Monitoring
        "enable_monitoring": True,
        "log_level": "INFO"
    }
    
    # Create deployment with daily schedule
    deployment = create_deployment(
        name="ev-charging-pipeline",
        schedule=CronSchedule(cron="0 3 * * *"),  # Daily at 3AM
        work_pool_name="default-agent-pool",
        config=ev_config
    )
    
    return deployment

def create_general_pipeline_deployment():
    """Create general purpose pipeline deployment"""
    
    # General configuration
    general_config = {
        "data_path": "data/general_data.json",
        "output_path": "outputs/general_pipeline",
        "model_name": "microsoft/DialoGPT-medium",
        "openai_api_key": os.getenv("OPENAI_API_KEY", "your-openai-api-key"),
        "domain": "general",
        "wandb_project": "general-llm-pipeline",
        "wandb_entity": os.getenv("WANDB_ENTITY", None),
        
        # Data processing
        "similarity_threshold": 0.8,
        "deduplication_method": "hybrid",
        "questions_per_doc": 3,
        "include_metadata": True,
        
        # Benchmark generation
        "benchmark_size": 100,
        "include_adversarial": True,
        "difficulty_levels": ["easy", "medium", "hard"],
        "categories": ["general", "technical", "business"],
        
        # Evaluation
        "evaluation_metrics": ["rouge", "bleu", "exact_match"],
        "domain_metrics": False,
        
        # Monitoring
        "enable_monitoring": True,
        "log_level": "INFO"
    }
    
    # Create deployment with weekly schedule
    deployment = create_deployment(
        name="general-llm-pipeline",
        schedule=CronSchedule(cron="0 2 * * 0"),  # Weekly on Sunday at 2AM
        work_pool_name="default-agent-pool",
        config=general_config
    )
    
    return deployment

def create_test_deployment():
    """Create test deployment without schedule"""
    
    # Test configuration
    test_config = {
        "data_path": "data/test_data.json",
        "output_path": "outputs/test_pipeline",
        "model_name": "microsoft/DialoGPT-medium",
        "openai_api_key": os.getenv("OPENAI_API_KEY", "your-openai-api-key"),
        "domain": "test",
        "wandb_project": "test-pipeline",
        "wandb_entity": os.getenv("WANDB_ENTITY", None),
        
        # Minimal processing for testing
        "similarity_threshold": 0.9,
        "deduplication_method": "levenshtein",
        "questions_per_doc": 2,
        "include_metadata": False,
        
        # Small benchmark for testing
        "benchmark_size": 20,
        "include_adversarial": False,
        "difficulty_levels": ["easy"],
        "categories": ["general"],
        
        # Basic evaluation
        "evaluation_metrics": ["rouge", "exact_match"],
        "domain_metrics": False,
        
        # Monitoring
        "enable_monitoring": True,
        "log_level": "DEBUG"
    }
    
    # Create deployment without schedule for manual testing
    deployment = create_deployment(
        name="test-pipeline",
        schedule=None,  # No schedule - manual execution only
        work_pool_name="default-agent-pool",
        config=test_config
    )
    
    return deployment

def setup_prefect_blocks():
    """Set up Prefect blocks for configuration management"""
    
    try:
        from prefect.blocks.system import Secret
        from prefect.filesystems import LocalFileSystem
        
        # Create local file system block
        local_fs = LocalFileSystem(
            basepath=os.path.abspath(".")
        )
        local_fs.save(name="pipeline-storage", overwrite=True)
        print("✅ Local file system block created")
        
        # Create secret blocks for API keys
        if os.getenv("OPENAI_API_KEY"):
            openai_secret = Secret(value=os.getenv("OPENAI_API_KEY"))
            openai_secret.save(name="openai-api-key", overwrite=True)
            print("✅ OpenAI API key secret created")
        
        if os.getenv("WANDB_API_KEY"):
            wandb_secret = Secret(value=os.getenv("WANDB_API_KEY"))
            wandb_secret.save(name="wandb-api-key", overwrite=True)
            print("✅ WandB API key secret created")
            
    except Exception as e:
        print(f"⚠️ Warning: Could not set up Prefect blocks: {e}")

def main():
    """Main deployment function"""
    
    print("🚀 Deploying LLM Pipeline with Prefect")
    print("=" * 50)
    
    # Set up Prefect blocks
    setup_prefect_blocks()
    
    # Create deployments
    deployments = []
    
    # Test deployment (no schedule)
    print("\n📋 Creating test deployment...")
    test_deployment = create_test_deployment()
    test_deployment.apply()
    deployments.append(test_deployment)
    print("✅ Test deployment created")
    
    # EV pipeline deployment
    print("\n🔋 Creating EV charging pipeline deployment...")
    ev_deployment = create_ev_pipeline_deployment()
    ev_deployment.apply()
    deployments.append(ev_deployment)
    print("✅ EV pipeline deployment created")
    
    # General pipeline deployment
    print("\n🌐 Creating general pipeline deployment...")
    general_deployment = create_general_pipeline_deployment()
    general_deployment.apply()
    deployments.append(general_deployment)
    print("✅ General pipeline deployment created")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Deployment Summary:")
    print(f"  Total deployments: {len(deployments)}")
    print(f"  Test deployment: {test_deployment.name}")
    print(f"  EV pipeline: {ev_deployment.name} (Daily at 3AM)")
    print(f"  General pipeline: {general_deployment.name} (Weekly on Sunday at 2AM)")
    
    print("\n🎯 Next Steps:")
    print("  1. Start Prefect server: prefect server start")
    print("  2. Start Prefect agent: prefect agent start -p default-agent-pool")
    print("  3. Monitor deployments: prefect deployment ls")
    print("  4. Run test deployment: prefect deployment run test-pipeline")
    
    print("\n📈 Monitoring:")
    print("  - Prefect UI: http://localhost:4200")
    print("  - Flow runs: prefect flow-run ls")
    print("  - Deployment logs: prefect deployment logs <deployment-name>")
    
    return deployments

if __name__ == "__main__":
    main() 