#!/usr/bin/env python3
"""
Server startup script for testing monitoring functionality
"""

import uvicorn
import logging
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from deployment.inference_server import app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Start the inference server"""
    logger.info("🚀 Starting LLM Inference Server with Monitoring")
    logger.info("=" * 50)
    
    # Server configuration
    host = "localhost"
    port = 8000
    reload = True
    
    logger.info(f"📍 Server will run on: http://{host}:{port}")
    logger.info("📊 Monitoring endpoints:")
    logger.info(f"  Health: http://{host}:{port}/health")
    logger.info(f"  Status: http://{host}:{port}/status")
    logger.info(f"  Prometheus: http://{host}:{port}/prometheus")
    logger.info(f"  Models: http://{host}:{port}/models")
    logger.info(f"  Inference: http://{host}:{port}/infer")
    logger.info("=" * 50)
    
    try:
        # Start the server
        uvicorn.run(
            "deployment.inference_server:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 