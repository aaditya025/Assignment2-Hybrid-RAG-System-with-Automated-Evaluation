"""
Text Chunking Module
Handles chunking of text into smaller pieces with overlap
"""

import json
from typing import List, Dict
from transformers import AutoTokenizer
import config


class TextChunker:
    def __init__(self, min_chunk_size=200, max_chunk_size=400, overlap=50):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        # Use a tokenizer for accurate token counting
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def chunk_text(self, text: str, url: str, title: str) -> List[Dict]:
        """
        Split text into overlapping chunks based on token count
        """
        # Tokenize the entire text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        chunks = []
        chunk_id = 0
        start_idx = 0

        while start_idx < len(tokens):
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.max_chunk_size, len(tokens))

            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            # Create chunk metadata
            chunk_data = {
                'chunk_id': f"{url}#chunk_{chunk_id}",
                'url': url,
                'title': title,
                'text': chunk_text,
                'token_count': len(chunk_tokens),
                'chunk_index': chunk_id
            }

            chunks.append(chunk_data)

            # Move to next chunk with overlap
            chunk_id += 1
            start_idx += (self.max_chunk_size - self.overlap)

        return chunks

    def chunk_multiple_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk multiple documents
        """
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_text(
                text=doc['text'],
                url=doc['url'],
                title=doc['title']
            )
            all_chunks.extend(chunks)

        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def save_chunks(self, chunks: List[Dict], filename: str):
        """
        Save chunks to JSON file
        """
        with open(filename, 'w') as f:
            json.dump(chunks, f, indent=2)
        print(f"Saved {len(chunks)} chunks to {filename}")

    def load_chunks(self, filename: str) -> List[Dict]:
        """
        Load chunks from JSON file
        """
        with open(filename, 'r') as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks from {filename}")
        return chunks


if __name__ == "__main__":
    # Test chunking
    chunker = TextChunker(
        min_chunk_size=config.MIN_CHUNK_SIZE,
        max_chunk_size=config.MAX_CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP
    )

    # Example document
    test_doc = {
        'url': 'https://test.com',
        'title': 'Test Document',
        'text': 'This is a test document. ' * 100
    }

    chunks = chunker.chunk_text(test_doc['text'], test_doc['url'], test_doc['title'])
    print(f"Created {len(chunks)} chunks")
