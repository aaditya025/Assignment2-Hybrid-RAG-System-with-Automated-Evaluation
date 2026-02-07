"""
Hybrid Retrieval Module
Implements Dense Vector Retrieval, BM25 Sparse Retrieval, and RRF
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
import pickle
import json
import os


class DenseRetriever:
    """Dense vector retrieval using sentence embeddings and FAISS"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = None
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension

    def build_index(self, chunks: List[Dict]):
        """
        Build FAISS index from chunks
        """
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]

        print("Encoding chunks with sentence transformer...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        # Create FAISS index
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))

        print(f"FAISS index built with {self.index.ntotal} vectors")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Retrieve top-k chunks for a query using dense retrieval
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # Return chunks with scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results

    def save_index(self, index_dir: str):
        """Save FAISS index and chunks"""
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(index_dir, 'faiss_index.bin'))
        with open(os.path.join(index_dir, 'chunks.pkl'), 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"Dense index saved to {index_dir}")

    def load_index(self, index_dir: str):
        """Load FAISS index and chunks"""
        self.index = faiss.read_index(os.path.join(index_dir, 'faiss_index.bin'))
        with open(os.path.join(index_dir, 'chunks.pkl'), 'rb') as f:
            self.chunks = pickle.load(f)
        print(f"Dense index loaded from {index_dir}")


class SparseRetriever:
    """BM25 sparse retrieval"""

    def __init__(self):
        self.bm25 = None
        self.chunks = None
        self.tokenized_corpus = None

    def build_index(self, chunks: List[Dict]):
        """
        Build BM25 index from chunks
        """
        self.chunks = chunks
        corpus = [chunk['text'] for chunk in chunks]

        # Simple tokenization
        self.tokenized_corpus = [doc.lower().split() for doc in corpus]

        print("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"BM25 index built with {len(self.tokenized_corpus)} documents")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Retrieve top-k chunks for a query using BM25
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Return chunks with scores
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))

        return results

    def save_index(self, index_dir: str):
        """Save BM25 index"""
        os.makedirs(index_dir, exist_ok=True)
        with open(os.path.join(index_dir, 'bm25_index.pkl'), 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'chunks': self.chunks,
                'tokenized_corpus': self.tokenized_corpus
            }, f)
        print(f"BM25 index saved to {index_dir}")

    def load_index(self, index_dir: str):
        """Load BM25 index"""
        with open(os.path.join(index_dir, 'bm25_index.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.chunks = data['chunks']
            self.tokenized_corpus = data['tokenized_corpus']
        print(f"BM25 index loaded from {index_dir}")


class HybridRetriever:
    """Hybrid retrieval combining dense and sparse methods with RRF"""

    def __init__(self, dense_retriever: DenseRetriever, sparse_retriever: SparseRetriever, rrf_k: int = 60):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[Dict, float]],
        sparse_results: List[Tuple[Dict, float]],
        top_n: int = 5
    ) -> List[Tuple[Dict, float, Dict]]:
        """
        Combine results using Reciprocal Rank Fusion
        RRF_score(d) = Σ 1/(k + rank_i(d)) where k=60
        """
        rrf_scores = {}
        chunk_data = {}
        retrieval_scores = {}

        # Process dense results
        for rank, (chunk, score) in enumerate(dense_results, start=1):
            chunk_id = chunk['chunk_id']
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + (1 / (self.rrf_k + rank))
            chunk_data[chunk_id] = chunk
            if chunk_id not in retrieval_scores:
                retrieval_scores[chunk_id] = {'dense': 0, 'sparse': 0}
            retrieval_scores[chunk_id]['dense'] = score

        # Process sparse results
        for rank, (chunk, score) in enumerate(sparse_results, start=1):
            chunk_id = chunk['chunk_id']
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + (1 / (self.rrf_k + rank))
            chunk_data[chunk_id] = chunk
            if chunk_id not in retrieval_scores:
                retrieval_scores[chunk_id] = {'dense': 0, 'sparse': 0}
            retrieval_scores[chunk_id]['sparse'] = score

        # Sort by RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top N
        results = []
        for chunk_id, rrf_score in sorted_chunks[:top_n]:
            results.append((
                chunk_data[chunk_id],
                rrf_score,
                retrieval_scores[chunk_id]
            ))

        return results

    def retrieve(self, query: str, top_k: int = 10, top_n: int = 5) -> Dict:
        """
        Perform hybrid retrieval
        Returns: dict with dense results, sparse results, and RRF combined results
        """
        # Dense retrieval
        dense_results = self.dense_retriever.retrieve(query, top_k)

        # Sparse retrieval
        sparse_results = self.sparse_retriever.retrieve(query, top_k)

        # RRF fusion
        rrf_results = self.reciprocal_rank_fusion(dense_results, sparse_results, top_n)

        return {
            'dense': dense_results,
            'sparse': sparse_results,
            'rrf': rrf_results
        }


if __name__ == "__main__":
    # Test retrieval
    print("Testing retrieval module...")
