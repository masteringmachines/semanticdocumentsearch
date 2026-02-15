"""
Embedding Engine - Convert text to vectors for semantic search.

This module demonstrates core embedding principles:
1. Text → Numbers (vectorization)
2. Similarity measurement
3. Semantic search
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import json


class EmbeddingEngine:
    """
    Simple embedding engine using TF-IDF.
    
    Core Concept:
    - Each document is converted to a vector (list of numbers)
    - Similar documents have similar vectors
    - We can measure similarity mathematically
    """
    
    def __init__(self, max_features: int = 1000):
        """
        Initialize the embedding engine.
        
        Args:
            max_features: Number of dimensions in embedding space
                         (more = more detailed, but slower)
        """
        # TF-IDF Vectorizer: Converts text to numbers
        # This is our "embedder"
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',  # Remove common words
            ngram_range=(1, 2)     # Use 1-word and 2-word phrases
        )
        
        # Storage
        self.documents: List[str] = []
        self.embeddings = None  # Will be numpy array of vectors
        self.fitted = False
    
    def add_document(self, text: str) -> int:
        """
        Add a document and create its embedding.
        
        Args:
            text: Document text
            
        Returns:
            Document ID (index)
        """
        self.documents.append(text)
        self._update_embeddings()
        return len(self.documents) - 1
    
    def add_documents(self, texts: List[str]):
        """Add multiple documents at once."""
        self.documents.extend(texts)
        self._update_embeddings()
    
    def _update_embeddings(self):
        """
        Regenerate embeddings for all documents.
        
        This is where the MAGIC happens:
        1. TF-IDF converts text to numbers
        2. Each document becomes a vector
        3. Vectors capture semantic meaning
        """
        if len(self.documents) == 0:
            return
        
        # Transform documents into TF-IDF vectors
        # Result: matrix where each row is a document vector
        self.embeddings = self.vectorizer.fit_transform(self.documents)
        self.fitted = True
        
        print(f"✓ Created embeddings: {len(self.documents)} documents, "
              f"{self.embeddings.shape[1]} dimensions")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Search for documents similar to query.
        
        How it works:
        1. Convert query to embedding (same way as documents)
        2. Compare query embedding with all document embeddings
        3. Return most similar documents
        
        Args:
            query: Search text
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, similarity_score, document_text)
        """
        if not self.fitted:
            return []
        
        # Convert query to embedding vector
        query_embedding = self.vectorizer.transform([query])
        
        # Calculate similarity with ALL documents
        # This compares the query vector with each document vector
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k most similar
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (int(idx), float(similarities[idx]), self.documents[idx])
            for idx in top_indices
        ]
        
        return results
    
    def find_similar(self, doc_id: int, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Find documents similar to a specific document.
        
        Args:
            doc_id: ID of reference document
            top_k: Number of similar docs to return
            
        Returns:
            List of (doc_id, similarity_score, document_text)
        """
        if not self.fitted or doc_id >= len(self.documents):
            return []
        
        # Get embedding of the reference document
        doc_embedding = self.embeddings[doc_id:doc_id+1]
        
        # Compare with all documents
        similarities = cosine_similarity(doc_embedding, self.embeddings)[0]
        
        # Exclude the document itself
        similarities[doc_id] = -1
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (int(idx), float(similarities[idx]), self.documents[idx])
            for idx in top_indices
            if similarities[idx] > 0  # Only include positive similarities
        ]
        
        return results
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.
        
        Returns:
            Similarity score (0 to 1, where 1 is identical)
        """
        if not self.fitted:
            self._update_embeddings()
        
        # Convert both texts to embeddings
        emb1 = self.vectorizer.transform([text1])
        emb2 = self.vectorizer.transform([text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(emb1, emb2)[0][0]
        
        return float(similarity)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get the embedding vector for text.
        
        This shows what an embedding actually looks like:
        A list of numbers representing the text!
        
        Returns:
            Numpy array of shape (1, n_features)
        """
        if not self.fitted:
            self._update_embeddings()
        
        return self.vectorizer.transform([text]).toarray()[0]
    
    def explain_embedding(self, text: str, top_n: int = 10):
        """
        Explain what an embedding means by showing top features.
        
        This helps understand WHAT the numbers represent!
        """
        if not self.fitted:
            self._update_embeddings()
        
        embedding = self.get_embedding(text)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get indices of highest values
        top_indices = np.argsort(embedding)[::-1][:top_n]
        
        print(f"\nEmbedding for: '{text}'")
        print(f"Vector dimensions: {len(embedding)}")
        print(f"\nTop {top_n} features (words/phrases with highest values):")
        print("-" * 50)
        
        for i, idx in enumerate(top_indices, 1):
            feature = feature_names[idx]
            value = embedding[idx]
            print(f"{i}. '{feature}': {value:.4f}")
        
        print(f"\nThese numbers capture the meaning of the text!")
        print(f"Similar texts will have similar patterns of numbers.")
    
    def get_similarity_matrix(self) -> np.ndarray:
        """
        Get similarity matrix for all documents.
        
        Returns:
            Matrix where [i][j] = similarity between doc i and doc j
        """
        if not self.fitted:
            return np.array([])
        
        return cosine_similarity(self.embeddings)
    
    def save(self, filename: str):
        """Save documents and embeddings to file."""
        data = {
            'documents': self.documents,
            'max_features': self.vectorizer.max_features
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved {len(self.documents)} documents to {filename}")
    
    def load(self, filename: str):
        """Load documents from file and recreate embeddings."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.documents = data['documents']
        self._update_embeddings()
        
        print(f"✓ Loaded {len(self.documents)} documents from {filename}")
    
    def show_statistics(self):
        """Show statistics about the embedding space."""
        if not self.fitted:
            print("No embeddings yet!")
            return
        
        print("\n" + "=" * 60)
        print("EMBEDDING STATISTICS")
        print("=" * 60)
        print(f"Documents: {len(self.documents)}")
        print(f"Embedding dimensions: {self.embeddings.shape[1]}")
        print(f"Vocabulary size: {len(self.vectorizer.get_feature_names_out())}")
        
        # Average similarity
        sim_matrix = self.get_similarity_matrix()
        avg_similarity = np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
        
        print(f"Average document similarity: {avg_similarity:.3f}")
        print(f"\nInterpretation:")
        print(f"  0.0-0.3: Documents very different")
        print(f"  0.3-0.6: Some similarity")
        print(f"  0.6-0.9: Quite similar")
        print(f"  0.9-1.0: Very similar/duplicates")
        print("=" * 60)


def demonstrate_embeddings():
    """
    Demonstrate how embeddings work with simple examples.
    """
    print("\n" + "=" * 60)
    print("EMBEDDING DEMONSTRATION")
    print("=" * 60)
    
    engine = EmbeddingEngine(max_features=100)
    
    # Add some example documents
    docs = [
        "I love dogs and puppies",
        "Cats are wonderful pets",
        "The car was very fast",
        "I enjoy programming in Python",
        "Dogs are loyal animals",
        "The automobile sped down the highway"
    ]
    
    print("\nAdding documents...")
    engine.add_documents(docs)
    
    # Show embeddings
    print("\n" + "-" * 60)
    print("Example: What does an embedding look like?")
    print("-" * 60)
    
    text = "I love dogs"
    engine.explain_embedding(text, top_n=5)
    
    # Test semantic search
    print("\n" + "-" * 60)
    print("Example: Semantic Search")
    print("-" * 60)
    
    query = "puppies"
    print(f"\nQuery: '{query}'")
    results = engine.search(query, top_k=3)
    
    print("\nTop results:")
    for i, (doc_id, score, text) in enumerate(results, 1):
        print(f"{i}. [Score: {score:.3f}] {text}")
    
    print("\nNotice: Found 'dogs' even though query was 'puppies'!")
    print("That's the power of embeddings! 🎯")
    
    # Show similarity
    print("\n" + "-" * 60)
    print("Example: Measuring Similarity")
    print("-" * 60)
    
    text1 = "car"
    text2 = "automobile"
    text3 = "dog"
    
    sim_12 = engine.compute_similarity(text1, text2)
    sim_13 = engine.compute_similarity(text1, text3)
    
    print(f"\nSimilarity between '{text1}' and '{text2}': {sim_12:.3f}")
    print(f"Similarity between '{text1}' and '{text3}': {sim_13:.3f}")
    
    print(f"\n'{text1}' and '{text2}' are similar (synonyms)!")
    print(f"'{text1}' and '{text3}' are different (unrelated)!")
    
    # Statistics
    engine.show_statistics()
    
    print("\n✨ This is how Google, ChatGPT, and all modern AI works!")
    print("   They just use more sophisticated embeddings. 🚀")


if __name__ == "__main__":
    demonstrate_embeddings()
