"""
Configuration settings for the Arabic Customer Service System
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "benchmark_results"
VECTORS_DIR = DATA_DIR / "vectors"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
VECTORS_DIR.mkdir(exist_ok=True)

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Model Configuration
AGENT_1_MODEL = os.getenv("AGENT_1_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
AGENT_2_MODEL = os.getenv("AGENT_2_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
AGENT_3_MODEL = os.getenv("AGENT_3_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")

# Dataset Paths
TEST_DATASET_PATH = DATA_DIR / "test_dataset.json"
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"

# Intent Categories
INTENT_CATEGORIES = [
    "product_inquiry",
    "technical_support",
    "billing_question",
    "complaint",
    "general_question",
    "escalation_needed"
]

# Knowledge Base Categories
KB_CATEGORIES = [
    "product_inquiry",
    "technical_support",
    "billing_question",
    "shipping",
    "general_question"
]

# ✅ UPDATED: Agent 2 Configuration with new embedding model
# ✅ UPDATED: Agent 2 Configuration with new embedding model
RETRIEVAL_CONFIG = {
    "n_results": 10,
    
    # ✅ FAST MODEL (only 80 MB - downloads in 1-2 minutes!)
    "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # ✅ Better for Arabic
    "collection_name": "arabic_knowledge_base",
    "persist_directory": str(VECTORS_DIR)
}
# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"