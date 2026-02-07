"""
Response Generator Module
Uses LLM to generate answers based on retrieved context
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict, Tuple


class ResponseGenerator:
    """Generate responses using LLM with retrieved context"""

    def __init__(self, model_name: str = "google/flan-t5-base", max_length: int = 512):
        print(f"Loading LLM model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
        # Load model with low memory settings
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.max_length = max_length

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")

    def create_prompt(self, query: str, context_chunks: List[Tuple[Dict, float, Dict]]) -> str:
        """
        Create prompt with query and retrieved context
        """
        # Extract chunk texts
        contexts = []
        for i, (chunk, rrf_score, scores) in enumerate(context_chunks, 1):
            contexts.append(f"[{i}] {chunk['text']}")

        context_text = "\n\n".join(contexts)

        # Create prompt
        prompt = f"""Answer the following question based on the provided context. If the answer cannot be found in the context, say "I don't have enough information to answer this question."

Context:
{context_text}

Question: {query}

Answer:"""

        return prompt

    def generate(self, query: str, context_chunks: List[Tuple[Dict, float, Dict]]) -> str:
        """
        Generate answer based on query and retrieved chunks
        """
        # Create prompt
        prompt = self.create_prompt(query, context_chunks)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=150,
                num_beams=4,
                early_stopping=True
            )

        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return answer

    def generate_with_metadata(
        self,
        query: str,
        retrieval_results: Dict
    ) -> Dict:
        """
        Generate answer and return with metadata
        """
        rrf_results = retrieval_results['rrf']

        # Generate answer
        answer = self.generate(query, rrf_results)

        # Prepare response with metadata
        response = {
            'query': query,
            'answer': answer,
            'retrieved_chunks': [],
            'dense_results': [],
            'sparse_results': []
        }

        # Add RRF results (final context used)
        for chunk, rrf_score, scores in rrf_results:
            response['retrieved_chunks'].append({
                'title': chunk['title'],
                'url': chunk['url'],
                'text': chunk['text'][:300] + '...',  # Truncate for display
                'rrf_score': round(rrf_score, 4),
                'dense_score': round(scores['dense'], 4),
                'sparse_score': round(scores['sparse'], 4)
            })

        # Add dense results for display
        for chunk, score in retrieval_results['dense'][:5]:
            response['dense_results'].append({
                'title': chunk['title'],
                'url': chunk['url'],
                'text': chunk['text'][:200] + '...',
                'score': round(score, 4)
            })

        # Add sparse results for display
        for chunk, score in retrieval_results['sparse'][:5]:
            response['sparse_results'].append({
                'title': chunk['title'],
                'url': chunk['url'],
                'text': chunk['text'][:200] + '...',
                'score': round(score, 4)
            })

        return response


if __name__ == "__main__":
    # Test generator
    generator = ResponseGenerator()
    print("Generator initialized successfully")
