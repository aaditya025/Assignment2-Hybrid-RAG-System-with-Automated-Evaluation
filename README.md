# Hybrid RAG System

A Retrieval-Augmented Generation (RAG) system that combines dense vector retrieval, sparse keyword retrieval (BM25), and Reciprocal Rank Fusion (RRF) to answer questions from Wikipedia articles.

## Features

- **Dense Vector Retrieval**: Uses sentence-transformers (all-MiniLM-L6-v2) with FAISS indexing for semantic similarity search
- **Sparse Keyword Retrieval**: BM25 algorithm for keyword-based retrieval
- **Reciprocal Rank Fusion (RRF)**: Combines results from both retrievers using RRF scoring
- **LLM Response Generation**: Uses Flan-T5-base for generating natural language answers
- **Interactive UI**: Streamlit web interface with detailed retrieval metrics

## System Architecture

```
Wikipedia Articles (500)
    ├── Fixed URLs (200) - Stored in fixed_urls.json
    └── Random URLs (300) - Generated per index build
            ↓
    Text Chunking (200-400 tokens, 50 overlap)
            ↓
    ┌───────────────────┬───────────────────┐
    │  Dense Retrieval  │ Sparse Retrieval  │
    │  (FAISS + Embed)  │      (BM25)       │
    └─────────┬─────────┴─────────┬─────────┘
              │                   │
              └─────────┬─────────┘
                        ↓
            Reciprocal Rank Fusion (RRF)
                        ↓
              LLM (Flan-T5-base)
                        ↓
                    Answer
```

## Installation

1. Clone the repository
```bash
cd HybridRAGSystem
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Build the System

First, build the indexes (this will take some time):

```bash
python build_index.py
```

This script will:
- Create or load 200 fixed Wikipedia URLs from `fixed_urls.json`
- Generate 300 random Wikipedia URLs
- Fetch and process all articles
- Create text chunks with overlap
- Build FAISS dense retrieval index
- Build BM25 sparse retrieval index

### 2. Run the Web Interface

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 3. Query the System

1. Enter your question in the text box
2. Adjust Top-K and Top-N parameters in the sidebar (optional)
3. Click "Generate Answer"
4. View results in different tabs:
   - **Answer**: The generated response
   - **Retrieved Chunks**: Top chunks used for answer generation with RRF scores
   - **Retrieval Details**: Separate dense and sparse retrieval results
   - **Metrics**: Performance metrics and system configuration

## Configuration

Edit `config.py` to customize:

- Chunk size and overlap
- Number of retrieved documents (Top-K, Top-N)
- RRF constant (k=60)
- Model names
- File paths

## Project Structure

```
HybridRAGSystem/
├── app.py                 # Streamlit web interface
├── build_index.py         # Index building pipeline
├── config.py              # Configuration settings
├── data_fetcher.py        # Wikipedia data fetching
├── text_chunker.py        # Text chunking module
├── retrieval.py           # Dense, Sparse, and Hybrid retrieval
├── generator.py           # LLM response generation
├── requirements.txt       # Python dependencies
├── fixed_urls.json        # Fixed Wikipedia URLs (200)
├── data/                  # Data directory
│   ├── documents.json     # Fetched Wikipedia articles
│   └── chunks.json        # Text chunks with metadata
└── indexes/               # Index files
    ├── faiss_index.bin    # FAISS vector index
    ├── chunks.pkl         # Chunks for dense retrieval
    └── bm25_index.pkl     # BM25 index
```

## Technical Details

### Dense Retrieval
- Model: sentence-transformers/all-MiniLM-L6-v2
- Embedding dimension: 384
- Similarity: Cosine similarity (via FAISS Inner Product with L2 normalization)

### Sparse Retrieval
- Algorithm: BM25 (Okapi BM25)
- Tokenization: Simple whitespace + lowercase

### Reciprocal Rank Fusion
- Formula: RRF_score(d) = Σ 1/(k + rank_i(d))
- k = 60 (default)

### Response Generation
- Model: google/flan-t5-base
- Context: Top-N chunks concatenated with query
- Generation: Beam search with temperature 0.7

## Requirements

- Python 3.8+
- 4GB+ RAM recommended
- GPU optional (will use CPU if not available)

## License

MIT License
