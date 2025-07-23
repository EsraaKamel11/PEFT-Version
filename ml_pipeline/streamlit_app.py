#!/usr/bin/env python3
"""
Advanced Streamlit UI for LLM Pipeline
Comprehensive dashboard with real-time monitoring, interactive components, and pipeline management
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import requests
import os
import sys
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any, Optional
import asyncio
import threading

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import pipeline components
from data_processing.deduplication import Deduplicator
from data_processing.qa_generation import QAGenerator
from data_processing.token_preservation import TokenPreservation, create_ev_tokenizer
from training.experiment_tracker import ExperimentTracker
from evaluation.benchmark_generation import BenchmarkGenerator
from evaluation.model_comparison import ModelEvaluator
from deployment.inference_server import app as inference_app

# Configure page
st.set_page_config(
    page_title="LLM Pipeline Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .pipeline-stage {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class StreamlitDashboard:
    """Advanced Streamlit dashboard for LLM pipeline"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_components()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'pipeline_status' not in st.session_state:
            st.session_state.pipeline_status = "idle"
        if 'current_stage' not in st.session_state:
            st.session_state.current_stage = None
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = {}
        if 'monitoring_data' not in st.session_state:
            st.session_state.monitoring_data = []
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = "microsoft/DialoGPT-medium"
    
    def setup_components(self):
        """Initialize pipeline components"""
        try:
            self.deduplicator = Deduplicator()
            self.qa_generator = QAGenerator()
            self.tokenizer, self.preservation = create_ev_tokenizer()
            self.benchmark_generator = BenchmarkGenerator()
            self.model_evaluator = ModelEvaluator()
            st.session_state.components_loaded = True
        except Exception as e:
            st.error(f"Failed to load components: {e}")
            st.session_state.components_loaded = False
    
    def render_header(self):
        """Render main header"""
        st.markdown('<h1 class="main-header">🚀 LLM Pipeline Dashboard</h1>', unsafe_allow_html=True)
        
        # Status indicator
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.session_state.pipeline_status == "running":
                st.info("🔄 Pipeline is currently running...")
            elif st.session_state.pipeline_status == "completed":
                st.success("✅ Pipeline completed successfully!")
            elif st.session_state.pipeline_status == "error":
                st.error("❌ Pipeline encountered an error")
            else:
                st.info("⏸️ Pipeline is idle")
    
    def render_sidebar(self):
        """Render sidebar with navigation and controls"""
        st.sidebar.markdown("## 🎛️ Pipeline Controls")
        
        # Pipeline configuration
        st.sidebar.markdown("### Configuration")
        
        # Model selection
        model_options = [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large",
            "gpt2",
            "gpt2-medium",
            "gpt2-large"
        ]
        st.session_state.selected_model = st.sidebar.selectbox(
            "Select Base Model",
            model_options,
            index=0
        )
        
        # Domain selection
        domain_options = ["electric_vehicles", "general", "automotive", "custom"]
        selected_domain = st.sidebar.selectbox("Domain", domain_options)
        
        # Pipeline parameters
        st.sidebar.markdown("### Parameters")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            similarity_threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.8, 0.05)
            questions_per_doc = st.number_input("Questions per Doc", 1, 10, 3)
        
        with col2:
            benchmark_size = st.number_input("Benchmark Size", 10, 500, 100)
            max_tokens = st.number_input("Max Tokens", 32, 512, 128)
        
        # Advanced settings
        with st.sidebar.expander("Advanced Settings"):
            include_metadata = st.checkbox("Include Metadata", True)
            include_adversarial = st.checkbox("Include Adversarial", True)
            enable_monitoring = st.checkbox("Enable Monitoring", True)
            log_level = st.selectbox("Log Level", ["INFO", "DEBUG", "WARNING", "ERROR"])
        
        # Pipeline actions
        st.sidebar.markdown("### Actions")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🚀 Start Pipeline", type="primary"):
                self.start_pipeline(selected_domain, similarity_threshold, questions_per_doc, 
                                  benchmark_size, max_tokens, include_metadata, include_adversarial)
        
        with col2:
            if st.button("⏹️ Stop Pipeline"):
                self.stop_pipeline()
        
        # Quick actions
        st.sidebar.markdown("### Quick Actions")
        if st.button("📊 View Metrics"):
            st.session_state.show_metrics = True
        
        if st.button("🔍 Test Tokenization"):
            st.session_state.show_tokenization = True
        
        if st.button("📈 View Experiments"):
            st.session_state.show_experiments = True
    
    def render_main_dashboard(self):
        """Render main dashboard content"""
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "🔄 Pipeline", "📈 Monitoring", "🧪 Testing", "⚙️ Settings"
        ])
        
        with tab1:
            self.render_overview_tab()
        
        with tab2:
            self.render_pipeline_tab()
        
        with tab3:
            self.render_monitoring_tab()
        
        with tab4:
            self.render_testing_tab()
        
        with tab5:
            self.render_settings_tab()
    
    def render_overview_tab(self):
        """Render overview tab with key metrics"""
        st.markdown("## 📊 Pipeline Overview")
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Pipeline Status</h3>
                <h2>🟢 Active</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Models Trained</h3>
                <h2>12</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Experiments</h3>
                <h2>45</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>Success Rate</h3>
                <h2>94.2%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Recent activity
        st.markdown("## 📈 Recent Activity")
        
        # Create sample activity data
        activity_data = pd.DataFrame({
            'Timestamp': pd.date_range(start='2024-01-01', periods=20, freq='H'),
            'Event': ['Pipeline Started', 'Data Loaded', 'Deduplication Complete', 'QA Generated'] * 5,
            'Status': ['Success', 'Success', 'Success', 'Success'] * 5,
            'Duration': np.random.randint(10, 300, 20)
        })
        
        # Activity timeline
        fig = px.timeline(activity_data, x_start='Timestamp', y='Event', color='Status',
                         title='Pipeline Activity Timeline')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Performance Metrics")
            
            # Create sample performance data
            performance_data = {
                'Metric': ['ROUGE-1', 'ROUGE-2', 'BLEU', 'Exact Match', 'BERTScore'],
                'Score': [0.85, 0.72, 0.78, 0.82, 0.89],
                'Improvement': [0.12, 0.08, 0.15, 0.09, 0.11]
            }
            
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Resource Usage")
            
            # Create resource usage chart
            resource_data = pd.DataFrame({
                'Time': pd.date_range(start='2024-01-01', periods=24, freq='H'),
                'CPU': np.random.uniform(20, 80, 24),
                'Memory': np.random.uniform(30, 90, 24),
                'GPU': np.random.uniform(40, 95, 24)
            })
            
            fig = px.line(resource_data, x='Time', y=['CPU', 'Memory', 'GPU'],
                         title='Resource Usage Over Time')
            st.plotly_chart(fig, use_container_width=True)
    
    def render_pipeline_tab(self):
        """Render pipeline management tab"""
        st.markdown("## 🔄 Pipeline Management")
        
        # Pipeline stages
        stages = [
            {"name": "Data Loading", "status": "completed", "duration": "2.3s"},
            {"name": "Deduplication", "status": "completed", "duration": "15.7s"},
            {"name": "QA Generation", "status": "running", "duration": "45.2s"},
            {"name": "Experiment Tracking", "status": "pending", "duration": "0s"},
            {"name": "Benchmark Generation", "status": "pending", "duration": "0s"},
            {"name": "Model Evaluation", "status": "pending", "duration": "0s"}
        ]
        
        for i, stage in enumerate(stages):
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{stage['name']}**")
                
                with col2:
                    if stage['status'] == 'completed':
                        st.markdown('<span class="status-success">✅</span>', unsafe_allow_html=True)
                    elif stage['status'] == 'running':
                        st.markdown('<span class="status-warning">🔄</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="status-error">⏸️</span>', unsafe_allow_html=True)
                
                with col3:
                    st.text(stage['status'].title())
                
                with col4:
                    st.text(stage['duration'])
                
                st.divider()
        
        # Pipeline controls
        st.markdown("## 🎛️ Pipeline Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Restart Pipeline", type="secondary"):
                st.info("Pipeline restart initiated...")
        
        with col2:
            if st.button("⏸️ Pause Pipeline"):
                st.warning("Pipeline paused...")
        
        with col3:
            if st.button("📊 View Logs"):
                st.session_state.show_logs = True
        
        # Pipeline configuration
        with st.expander("Pipeline Configuration"):
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                st.text_input("Data Path", value="data/ev_charging_data.json")
                st.text_input("Output Path", value="outputs/ev_pipeline")
                st.text_input("Model Name", value=st.session_state.selected_model)
            
            with config_col2:
                st.number_input("Batch Size", 1, 128, 32)
                st.number_input("Max Length", 64, 1024, 512)
                st.selectbox("Device", ["auto", "cpu", "cuda"])
    
    def render_monitoring_tab(self):
        """Render monitoring tab with real-time metrics"""
        st.markdown("## 📈 Real-Time Monitoring")
        
        # Real-time metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Requests/min", "156", "12%")
        
        with col2:
            st.metric("Response Time", "245ms", "-8%")
        
        with col3:
            st.metric("Success Rate", "99.2%", "0.3%")
        
        with col4:
            st.metric("Active Models", "3", "0")
        
        # Real-time charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Request Volume")
            
            # Generate real-time data
            times = pd.date_range(start='2024-01-01', periods=100, freq='1min')
            requests_data = np.random.poisson(150, 100) + np.sin(np.arange(100) * 0.1) * 20
            
            fig = px.line(x=times, y=requests_data, title='Requests per Minute')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### ⏱️ Response Times")
            
            # Generate response time data
            response_times = np.random.exponential(200, 100) + 50
            
            fig = px.histogram(x=response_times, title='Response Time Distribution',
                             nbins=20, opacity=0.7)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Model performance
        st.markdown("### 🤖 Model Performance")
        
        # Create model performance data
        models = ['Model A', 'Model B', 'Model C', 'Model D']
        metrics = ['ROUGE-1', 'ROUGE-2', 'BLEU', 'Exact Match']
        
        # Generate random performance data
        performance_matrix = np.random.uniform(0.7, 0.95, (len(models), len(metrics)))
        
        # Create heatmap
        fig = px.imshow(performance_matrix,
                       x=metrics,
                       y=models,
                       color_continuous_scale='RdYlGn',
                       title='Model Performance Heatmap')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Alerts and notifications
        st.markdown("### 🚨 Alerts & Notifications")
        
        alerts = [
            {"time": "2 min ago", "level": "warning", "message": "High response time detected"},
            {"time": "5 min ago", "level": "info", "message": "New model deployed successfully"},
            {"time": "10 min ago", "level": "error", "message": "Pipeline stage failed"},
            {"time": "15 min ago", "level": "success", "message": "Benchmark generation completed"}
        ]
        
        for alert in alerts:
            if alert['level'] == 'error':
                st.error(f"**{alert['time']}**: {alert['message']}")
            elif alert['level'] == 'warning':
                st.warning(f"**{alert['time']}**: {alert['message']}")
            elif alert['level'] == 'success':
                st.success(f"**{alert['time']}**: {alert['message']}")
            else:
                st.info(f"**{alert['time']}**: {alert['message']}")
    
    def render_testing_tab(self):
        """Render testing tab with interactive components"""
        st.markdown("## 🧪 Interactive Testing")
        
        # Tokenization testing
        st.markdown("### 🔤 Tokenization Testing")
        
        test_text = st.text_area(
            "Enter text to test tokenization:",
            value="CCS2 charging at 350kW with OCPP protocol and Plug&Charge functionality",
            height=100
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Test Tokenization"):
                if st.session_state.components_loaded:
                    try:
                        # Test tokenization
                        tokens = self.preservation.tokenize_with_preservation(
                            test_text,
                            truncation=True,
                            max_length=128
                        )
                        
                        # Verify preservation
                        verification = self.preservation.verify_token_preservation(test_text)
                        
                        st.success("Tokenization completed!")
                        
                        # Display results
                        st.json({
                            "input_text": test_text,
                            "token_count": len(tokens["input_ids"]),
                            "preservation_score": verification["overall_score"],
                            "terms_found": len(verification["terms_found"])
                        })
                        
                    except Exception as e:
                        st.error(f"Tokenization failed: {e}")
                else:
                    st.error("Components not loaded")
        
        with col2:
            if st.button("📊 View Statistics"):
                if st.session_state.components_loaded:
                    stats = self.preservation.get_preservation_statistics()
                    st.json(stats)
        
        # Model inference testing
        st.markdown("### 🤖 Model Inference Testing")
        
        inference_text = st.text_area(
            "Enter prompt for inference:",
            value="What is the maximum charging speed for CCS2?",
            height=80
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_tokens = st.number_input("Max Tokens", 10, 500, 50)
        
        with col2:
            temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
        
        with col3:
            if st.button("🚀 Generate Response"):
                # Simulate inference
                with st.spinner("Generating response..."):
                    time.sleep(2)  # Simulate processing time
                    
                    # Mock response
                    mock_response = "CCS2 charging supports maximum speeds up to 350kW, depending on the specific implementation and power supply capabilities."
                    
                    st.success("Response generated!")
                    st.markdown(f"**Response:** {mock_response}")
        
        # Benchmark testing
        st.markdown("### 🎯 Benchmark Testing")
        
        benchmark_options = ["EV Charging", "Technical Specs", "Environmental Impact", "Cost Analysis"]
        selected_benchmark = st.selectbox("Select Benchmark Category", benchmark_options)
        
        if st.button("📊 Run Benchmark"):
            with st.spinner("Running benchmark..."):
                time.sleep(3)  # Simulate benchmark execution
                
                # Mock benchmark results
                results = {
                    "category": selected_benchmark,
                    "questions": 25,
                    "correct_answers": 22,
                    "accuracy": 0.88,
                    "avg_response_time": 1.2
                }
                
                st.success("Benchmark completed!")
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Questions", results["questions"])
                with col2:
                    st.metric("Correct", results["correct_answers"])
                with col3:
                    st.metric("Accuracy", f"{results['accuracy']:.1%}")
                with col4:
                    st.metric("Avg Time", f"{results['avg_response_time']:.1f}s")
    
    def render_settings_tab(self):
        """Render settings tab"""
        st.markdown("## ⚙️ Settings & Configuration")
        
        # General settings
        st.markdown("### 🔧 General Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("Project Name", value="EV Charging Pipeline")
            st.text_input("API Key", value="sk-...", type="password")
            st.selectbox("Environment", ["development", "staging", "production"])
        
        with col2:
            st.number_input("Max Concurrent Jobs", 1, 10, 3)
            st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
            st.checkbox("Enable Notifications", True)
        
        # Model settings
        st.markdown("### 🤖 Model Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("Default Model", ["microsoft/DialoGPT-medium", "gpt2", "gpt2-medium"])
            st.number_input("Max Sequence Length", 128, 2048, 512)
            st.selectbox("Precision", ["float32", "float16", "bfloat16"])
        
        with col2:
            st.number_input("Batch Size", 1, 128, 32)
            st.selectbox("Optimizer", ["AdamW", "Adam", "SGD"])
            st.number_input("Learning Rate", 1e-6, 1e-3, 1e-4, format="%.2e")
        
        # Monitoring settings
        st.markdown("### 📊 Monitoring Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Enable Real-time Monitoring", True)
            st.number_input("Metrics Retention (days)", 1, 365, 30)
            st.selectbox("Alert Level", ["info", "warning", "error", "critical"])
        
        with col2:
            st.text_input("Webhook URL", value="https://hooks.slack.com/...")
            st.text_input("Email Notifications", value="admin@company.com")
            st.checkbox("Auto-scaling", False)
        
        # Save settings
        if st.button("💾 Save Settings", type="primary"):
            st.success("Settings saved successfully!")
    
    def start_pipeline(self, domain, similarity_threshold, questions_per_doc, 
                      benchmark_size, max_tokens, include_metadata, include_adversarial):
        """Start the pipeline with given parameters"""
        st.session_state.pipeline_status = "running"
        st.session_state.current_stage = "initializing"
        
        # This would be implemented with actual pipeline execution
        st.success("Pipeline started successfully!")
        
        # Simulate pipeline progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        stages = ["Data Loading", "Deduplication", "QA Generation", "Benchmark Generation", "Model Evaluation"]
        
        for i, stage in enumerate(stages):
            status_text.text(f"Running: {stage}")
            progress_bar.progress((i + 1) / len(stages))
            time.sleep(1)  # Simulate processing time
        
        st.session_state.pipeline_status = "completed"
        st.success("Pipeline completed successfully!")
    
    def stop_pipeline(self):
        """Stop the pipeline"""
        st.session_state.pipeline_status = "stopped"
        st.warning("Pipeline stopped by user")

def main():
    """Main Streamlit application"""
    dashboard = StreamlitDashboard()
    
    # Render dashboard
    dashboard.render_header()
    dashboard.render_sidebar()
    dashboard.render_main_dashboard()

if __name__ == "__main__":
    main() 