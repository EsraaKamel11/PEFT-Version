# 🚀 Prefect Scheduling for LLM Pipeline

Comprehensive scheduling and orchestration system for the LLM fine-tuning pipeline using Prefect.

## 📋 Overview

The Prefect scheduling system provides:

- ✅ **Automated Pipeline Execution** with cron schedules
- ✅ **Error Handling & Recovery** with retry mechanisms
- ✅ **Real-time Monitoring** with Prefect UI
- ✅ **Configuration Management** with environment variables
- ✅ **Multiple Deployment Types** (test, production, domain-specific)
- ✅ **Resource Management** with work pools
- ✅ **Logging & Metrics** with structured logging

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prefect UI    │    │  Prefect Server │    │  Work Pool      │
│   (Monitoring)  │◄──►│   (Orchestrator)│◄──►│   (Executors)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │    Pipeline Stages      │
                    │  ┌─────────────────────┐ │
                    │  │ 1. Data Loading     │ │
                    │  │ 2. Deduplication    │ │
                    │  │ 3. QA Generation    │ │
                    │  │ 4. Experiment Track │ │
                    │  │ 5. Benchmark Gen    │ │
                    │  │ 6. Model Evaluation │ │
                    │  │ 7. Finalization     │ │
                    │  └─────────────────────┘ │
                    └─────────────────────────┘
```

## 🔧 Quick Start

### 1. Install Prefect

```bash
pip install prefect
```

### 2. Set Environment Variables

```bash
# API Keys
export OPENAI_API_KEY="your-openai-api-key"
export WANDB_API_KEY="your-wandb-api-key"
export WANDB_ENTITY="your-wandb-entity"

# Pipeline Configuration
export PIPELINE_ENV="production"
export LOG_LEVEL="INFO"
```

### 3. Deploy Pipeline

```bash
cd ml_pipeline
python deploy_pipeline.py
```

### 4. Start Prefect Services

```bash
# Start Prefect server
prefect server start

# In another terminal, start work pool agent
prefect agent start -p default-agent-pool
```

### 5. Monitor Pipeline

- **Prefect UI**: http://localhost:4200
- **Flow Runs**: `prefect flow-run ls`
- **Deployment Logs**: `prefect deployment logs <deployment-name>`

## 📅 Deployment Types

### 1. **Test Deployment**
- **Schedule**: Manual execution only
- **Purpose**: Development and testing
- **Configuration**: Minimal processing, small datasets
- **Usage**: `prefect deployment run test-pipeline`

### 2. **EV Charging Pipeline**
- **Schedule**: Daily at 3:00 AM (`0 3 * * *`)
- **Purpose**: Electric vehicle domain-specific processing
- **Features**: 
  - Domain-specific metrics
  - Adversarial testing
  - Comprehensive evaluation
- **Categories**: Pricing, Technical, Compatibility, Environmental

### 3. **General Pipeline**
- **Schedule**: Weekly on Sunday at 2:00 AM (`0 2 * * 0`)
- **Purpose**: General-purpose LLM processing
- **Features**:
  - Standard evaluation metrics
  - Balanced processing
  - Weekly updates

## ⚙️ Configuration

### **Environment Variables**

```bash
# Required
OPENAI_API_KEY=sk-...
WANDB_API_KEY=wandb_...

# Optional
WANDB_ENTITY=your-username
PIPELINE_ENV=production
LOG_LEVEL=INFO
```

### **Pipeline Configuration**

```python
# EV Pipeline Configuration
ev_config = {
    "data_path": "data/ev_charging_data.json",
    "output_path": "outputs/ev_pipeline",
    "model_name": "microsoft/DialoGPT-medium",
    "domain": "electric_vehicles",
    "wandb_project": "ev-charging-pipeline",
    
    # Data Processing
    "similarity_threshold": 0.85,
    "deduplication_method": "hybrid",
    "questions_per_doc": 5,
    "include_metadata": True,
    
    # Benchmark Generation
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
```

### **Schedule Configuration**

```python
from prefect.server.schemas.schedules import CronSchedule

# Daily at 3 AM
daily_schedule = CronSchedule(cron="0 3 * * *")

# Weekly on Sunday at 2 AM
weekly_schedule = CronSchedule(cron="0 2 * * 0")

# Every 6 hours
frequent_schedule = CronSchedule(cron="0 */6 * * *")

# Weekdays only at 9 AM
weekday_schedule = CronSchedule(cron="0 9 * * 1-5")
```

## 📊 Monitoring & Observability

### **Prefect UI Dashboard**

Access the Prefect UI at `http://localhost:4200` to monitor:

- **Flow Runs**: Real-time execution status
- **Deployments**: Scheduled pipeline configurations
- **Logs**: Detailed execution logs
- **Metrics**: Performance and error metrics
- **Artifacts**: Generated outputs and reports

### **Key Metrics**

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Flow Run Duration | Total execution time | > 2 hours |
| Success Rate | Successful runs percentage | < 95% |
| Error Rate | Failed runs percentage | > 5% |
| Queue Time | Time in queue before execution | > 30 minutes |
| Resource Usage | CPU/Memory utilization | > 80% |

### **Logging Levels**

```python
# Configure logging levels
config = {
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "enable_monitoring": True,
    "structured_logging": True
}
```

## 🔄 Pipeline Stages

### **1. Initialize Pipeline**
```python
@task(name="initialize-pipeline")
def initialize_pipeline(self) -> Dict[str, Any]:
    """Initialize pipeline with configuration and monitoring"""
    # Validate configuration
    # Create output directories
    # Set up monitoring
    # Return initialization status
```

### **2. Load and Deduplicate Data**
```python
@task(name="load-and-deduplicate-data")
def load_and_deduplicate_data(self, init_result: Dict[str, Any]) -> Dict[str, Any]:
    """Load and deduplicate input data"""
    # Load data from source
    # Apply deduplication
    # Save processed data
    # Return statistics
```

### **3. Generate QA Pairs**
```python
@task(name="generate-qa-pairs")
def generate_qa_pairs(self, dedup_result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate QA pairs from deduplicated data"""
    # Initialize QA generator
    # Generate questions and answers
    # Include metadata
    # Save QA pairs
```

### **4. Setup Experiment Tracking**
```python
@task(name="setup-experiment-tracking")
def setup_experiment_tracking(self, qa_result: Dict[str, Any]) -> Dict[str, Any]:
    """Set up experiment tracking with WandB"""
    # Initialize WandB
    # Log configuration
    # Log initial metrics
    # Return experiment info
```

### **5. Generate Benchmark**
```python
@task(name="generate-benchmark")
def generate_benchmark(self, tracking_result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate benchmark dataset"""
    # Create benchmark generator
    # Generate domain-specific questions
    # Include adversarial examples
    # Save benchmark dataset
```

### **6. Evaluate Models**
```python
@task(name="evaluate-models")
def evaluate_models(self, benchmark_result: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate models using benchmark"""
    # Load models
    # Run evaluation
    # Calculate metrics
    # Generate comparison report
```

### **7. Finalize Pipeline**
```python
@task(name="finalize-pipeline")
def finalize_pipeline(self, eval_result: Dict[str, Any]) -> Dict[str, Any]:
    """Finalize pipeline and generate summary"""
    # Calculate final metrics
    # Generate summary report
    # Clean up resources
    # Return final status
```

## 🚨 Error Handling & Recovery

### **Retry Configuration**

```python
@task(
    name="generate-qa-pairs",
    retries=3,
    retry_delay_seconds=60,
    retry_jitter_factor=0.1
)
def generate_qa_pairs(self, dedup_result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate QA pairs with retry logic"""
    try:
        # QA generation logic
        pass
    except Exception as e:
        # Log error and retry
        self.logger.error(f"QA generation failed: {e}")
        raise
```

### **Error Types & Handling**

| Error Type | Handling Strategy | Retry Count |
|------------|-------------------|-------------|
| API Rate Limits | Exponential backoff | 5 |
| Network Timeouts | Linear backoff | 3 |
| Model Loading | Immediate retry | 2 |
| Data Validation | No retry | 0 |
| Resource Exhaustion | Long delay | 1 |

### **Alerting Configuration**

```python
# Configure alerts for critical failures
alerts = {
    "pipeline_failure": {
        "condition": "flow_run.state == 'Failed'",
        "notification": "slack://channel/pipeline-alerts",
        "cooldown": "1 hour"
    },
    "high_error_rate": {
        "condition": "error_rate > 0.1",
        "notification": "email://admin@company.com",
        "cooldown": "30 minutes"
    }
}
```

## 🔧 Advanced Configuration

### **Custom Schedules**

```python
# Complex scheduling patterns
schedules = {
    # Business hours only
    "business_hours": CronSchedule(cron="0 9-17 * * 1-5"),
    
    # Weekend processing
    "weekend": CronSchedule(cron="0 2 * * 6,0"),
    
    # Monthly processing
    "monthly": CronSchedule(cron="0 1 1 * *"),
    
    # Quarterly processing
    "quarterly": CronSchedule(cron="0 1 1 */3 *")
}
```

### **Resource Management**

```python
# Configure work pool resources
work_pool_config = {
    "name": "llm-pipeline-pool",
    "work_queues": [
        {
            "name": "high-priority",
            "priority": 1,
            "concurrency_limit": 2
        },
        {
            "name": "normal",
            "priority": 2,
            "concurrency_limit": 5
        },
        {
            "name": "low-priority",
            "priority": 3,
            "concurrency_limit": 10
        }
    ]
}
```

### **Dependencies & DAG**

```python
# Define task dependencies
@flow(name="LLM Pipeline")
def main_pipeline_flow(config: Dict[str, Any]) -> Dict[str, Any]:
    # Sequential execution
    init_result = initialize_pipeline()
    dedup_result = load_and_deduplicate_data(init_result)
    qa_result = generate_qa_pairs(dedup_result)
    tracking_result = setup_experiment_tracking(qa_result)
    benchmark_result = generate_benchmark(tracking_result)
    eval_result = evaluate_models(benchmark_result)
    final_result = finalize_pipeline(eval_result)
    
    return final_result
```

## 📈 Performance Optimization

### **Parallel Processing**

```python
# Parallel task execution
@flow(name="Parallel Pipeline")
def parallel_pipeline_flow(config: Dict[str, Any]) -> Dict[str, Any]:
    # Run independent tasks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        dedup_future = executor.submit(load_and_deduplicate_data, init_result)
        qa_future = executor.submit(generate_qa_pairs, dedup_result)
        benchmark_future = executor.submit(generate_benchmark, tracking_result)
    
    # Wait for completion
    dedup_result = dedup_future.result()
    qa_result = qa_future.result()
    benchmark_result = benchmark_future.result()
```

### **Caching Strategy**

```python
# Cache expensive operations
@task(cache_key_fn=lambda context: context.parameters["data_hash"])
def expensive_computation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Cache expensive computations"""
    # Computation logic
    pass
```

### **Resource Scaling**

```python
# Auto-scaling configuration
scaling_config = {
    "min_workers": 1,
    "max_workers": 10,
    "scale_up_threshold": 0.8,
    "scale_down_threshold": 0.2,
    "scale_up_cooldown": "5 minutes",
    "scale_down_cooldown": "10 minutes"
}
```

## 🔍 Troubleshooting

### **Common Issues**

1. **Prefect Server Not Starting**
   ```bash
   # Check if port is in use
   netstat -an | grep 4200
   
   # Kill existing process
   pkill -f prefect
   
   # Start server
   prefect server start
   ```

2. **Work Pool Agent Not Running**
   ```bash
   # Check agent status
   prefect agent ls
   
   # Start agent
   prefect agent start -p default-agent-pool
   ```

3. **Deployment Not Executing**
   ```bash
   # Check deployment status
   prefect deployment ls
   
   # Check work queue
   prefect work-queue ls
   
   # Manually trigger
   prefect deployment run <deployment-name>
   ```

### **Debug Commands**

```bash
# View flow runs
prefect flow-run ls

# View deployment logs
prefect deployment logs <deployment-name>

# View task logs
prefect task-run ls

# Check server health
prefect server health

# View work pool status
prefect work-pool ls
```

## 📚 Additional Resources

- [Prefect Documentation](https://docs.prefect.io/)
- [Cron Schedule Generator](https://crontab.guru/)
- [Prefect UI Guide](https://docs.prefect.io/ui/)
- [Work Pool Configuration](https://docs.prefect.io/concepts/work-pools/)
- [Deployment Best Practices](https://docs.prefect.io/concepts/deployments/)

---

**🎉 Your LLM pipeline is now fully scheduled and production-ready with Prefect!** 