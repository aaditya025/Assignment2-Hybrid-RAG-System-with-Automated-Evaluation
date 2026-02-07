"""
Configuration file for Hybrid RAG System
"""

# Dataset Configuration
FIXED_URLS_COUNT = 200
RANDOM_URLS_COUNT = 300
TOTAL_URLS = 500
MIN_WORDS_PER_PAGE = 200

# Chunking Configuration
MIN_CHUNK_SIZE = 200  # tokens
MAX_CHUNK_SIZE = 400  # tokens
CHUNK_OVERLAP = 50    # tokens

# Retrieval Configuration
TOP_K_DENSE = 10      # Top K chunks from dense retrieval
TOP_K_SPARSE = 10     # Top K chunks from sparse retrieval
TOP_N_FINAL = 5       # Top N chunks after RRF for context
RRF_K = 60            # RRF constant

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"

# File Paths
FIXED_URLS_FILE = "fixed_urls.json"
DATA_DIR = "data"
INDEX_DIR = "indexes"
CHUNKS_FILE = "data/chunks.json"
VECTOR_INDEX_FILE = "indexes/faiss_index"
