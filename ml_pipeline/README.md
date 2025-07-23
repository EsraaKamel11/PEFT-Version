# EV Charging Stations LLM Fine-tuning Pipeline

An end-to-end pipeline for fine-tuning language models on electric vehicle charging station data, with automated data collection, processing, training, evaluation, and deployment.

## 🚀 Features

- **Automated Data Collection**: Web scraping and PDF extraction for EV charging data
- **Intelligent Processing**: Cleaning, deduplication, and quality filtering
- **QA Generation**: Automated question-answer pair creation using LLM APIs
- **Memory-Efficient Training**: QLoRA fine-tuning for 7B parameter models
- **Comprehensive Evaluation**: Domain-specific benchmarks and metrics
- **Production Deployment**: FastAPI server with authentication and monitoring
- **MLOps Integration**: CI/CD, orchestration, and monitoring

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (for training)
- OpenAI API key (for QA generation)
- HuggingFace access token (for model downloads)

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd ml_pipeline
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Install Playwright browsers**:
```bash
playwright install
```

## 🚀 Quick Start

### 1. Configuration
Edit `config/config.yaml` to customize:
- Domain-specific URLs
- Model parameters
- Training settings
- Evaluation metrics

### 2. Run the Complete Pipeline
```bash
python main.py
```

This will execute:
- Data collection from EV charging websites
- Data processing and cleaning
- QA pair generation
- Model fine-tuning with QLoRA
- Evaluation and benchmarking
- Model registration for deployment

### 3. Deploy the Model
```bash
# Start the inference server
uvicorn src.deployment.inference_server:app --host 0.0.0.0 --port 8000

# Test the API
curl -X POST "http://localhost:8000/infer" \
  -H "api-key: test-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is Level 2 charging?",
    "adapter_version": "v1.0",
    "base_model": "meta-llama/Llama-2-7b-hf"
  }'
```

## 📁 Project Structure

```
ml_pipeline/
├── config/                 # Configuration files
│   ├── settings.py        # Pydantic settings with env support
│   └── config.yaml        # Domain-specific configuration
├── src/
│   ├── data_collection/   # Web scraping, PDF extraction
│   ├── data_processing/   # Cleaning, filtering, storage
│   ├── training/          # Model loading, LoRA, training loop
│   ├── evaluation/        # Benchmarks, metrics, comparison
│   ├── deployment/        # FastAPI server, model registry
│   └── orchestration/     # Workflow automation
├── tests/                 # Unit and integration tests
├── docker/               # Containerization files
├── main.py               # End-to-end pipeline
└── requirements.txt      # Dependencies
```

## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=sk-...          # For QA generation
WANDB_API_KEY=...              # For experiment tracking
SLACK_API_TOKEN=...            # For failure alerts
SLACK_CHANNEL=#general         # Slack channel for alerts
```

### Model Configuration
- **Base Model**: Llama-2-7B (configurable)
- **Fine-tuning**: QLoRA with rank 16, alpha 32
- **Quantization**: 4-bit for memory efficiency
- **Training**: Gradient accumulation, mixed precision

## 📊 Evaluation

The pipeline includes comprehensive evaluation:

- **Domain-specific benchmarks**: Auto-generated EV charging questions
- **Standard metrics**: ROUGE, BLEU, F1, Exact Match
- **Performance metrics**: Latency, throughput
- **Baseline comparison**: Fine-tuned vs base model

## 🚀 Deployment

### Docker Deployment
```bash
# Build and run with Docker
docker-compose -f docker/docker-compose.yml up --build
```

### Production Considerations
- Set up proper API keys and authentication
- Configure monitoring and alerting
- Use GPU-enabled containers for inference
- Implement rate limiting and security

## 🧪 Testing

Run the test suite:
```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/test_integration.py

# Full pipeline test (requires API keys)
python main.py
```

## 📈 Monitoring

- **Experiment Tracking**: Weights & Biases integration
- **API Monitoring**: Prometheus metrics
- **Pipeline Monitoring**: Slack alerts for failures
- **Performance Monitoring**: Latency and throughput tracking

## 🔄 CI/CD

The pipeline includes GitHub Actions for:
- Automated testing
- Code quality checks
- Docker image building
- Deployment automation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions:
- Check the documentation
- Review the test suite
- Open an issue on GitHub

## 🎯 Domain Customization

To adapt this pipeline for other domains:

1. Update `config/config.yaml` with domain-specific URLs and prompts
2. Modify the QA generation prompts in `src/training/dataset_preparation.py`
3. Update benchmark questions in `src/evaluation/benchmark_creator.py`
4. Adjust data collection URLs in `main.py`

## 📊 Performance

Typical performance metrics:
- **Training Time**: 2-4 hours on RTX 4090
- **Inference Latency**: <500ms per request
- **Memory Usage**: ~8GB GPU memory during training
- **Model Size**: ~4GB (4-bit quantized)

## 🔮 Future Enhancements

- Multi-modal support (images, diagrams)
- Real-time data collection
- Advanced retrieval-augmented generation
- A/B testing framework
- Auto-scaling deployment 