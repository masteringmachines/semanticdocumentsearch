#!/usr/bin/env python3
"""
Semantic Search Engine - Main Interface
Learn embeddings by searching documents!
"""

import sys
from src.embeddings import EmbeddingEngine, demonstrate_embeddings


# Sample documents for demo
SAMPLE_DOCS = [
    "Climate change is causing polar ice caps to melt at an alarming rate.",
    "Global warming leads to rising sea levels and extreme weather events.",
    "Artificial intelligence is transforming the technology industry.",
    "Machine learning algorithms can recognize patterns in large datasets.",
    "The stock market experienced significant volatility yesterday.",
    "Investors are concerned about economic recession indicators.",
    "Regular exercise improves cardiovascular health and fitness.",
    "A balanced diet with fruits and vegetables promotes wellness.",
    "Python is a popular programming language for data science.",
    "JavaScript frameworks make web development more efficient.",
    "Renewable energy sources like solar and wind are becoming cheaper.",
    "Electric vehicles are replacing traditional gasoline-powered cars.",
    "Space exploration missions are discovering new exoplanets.",
    "NASA's Mars rover is searching for signs of ancient life.",
    "Online education platforms provide accessible learning opportunities.",
    "Remote work has become common in many industries.",
]


def demo_mode():
    """Run interactive demo showing embedding concepts."""
    print("\n" + "=" * 70)
    print("SEMANTIC SEARCH ENGINE - DEMO MODE")
    print("=" * 70)
    print("\nThis demo shows how embeddings enable semantic search!")
    print("\nWhat are embeddings?")
    print("→ Embeddings convert text into numbers (vectors)")
    print("→ Similar meanings have similar numbers")
    print("→ This enables searching by MEANING, not just keywords")
    
    # Create engine and add documents
    print("\n" + "-" * 70)
    print("Loading sample documents...")
    print("-" * 70)
    
    engine = EmbeddingEngine()
    engine.add_documents(SAMPLE_DOCS)
    
    print(f"\n✓ Loaded {len(SAMPLE_DOCS)} documents")
    print(f"✓ Each document is now a vector with {engine.embeddings.shape[1]} dimensions")
    
    # Show statistics
    engine.show_statistics()
    
    # Interactive search
    print("\n" + "=" * 70)
    print("SEMANTIC SEARCH DEMO")
    print("=" * 70)
    print("\nTry searching! The engine understands meaning, not just keywords.")
    print("Examples: 'global warming', 'AI technology', 'health fitness'")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            query = input("Search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            # Search
            results = engine.search(query, top_k=3)
            
            print(f"\n🔍 Searching for: '{query}'")
            print("-" * 70)
            
            if results:
                print(f"\nTop {len(results)} results:")
                for i, (doc_id, score, text) in enumerate(results, 1):
                    print(f"\n{i}. [Score: {score:.3f}]")
                    print(f"   {text}")
                
                # Explain why it matched
                print(f"\n💡 Why these results?")
                print(f"   The query embedding is similar to these document embeddings.")
                print(f"   Higher scores = more similar meanings!")
            else:
                print("\nNo results found.")
            
            print()
            
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n")
            break
    
    print("\n✨ That's semantic search with embeddings!")
    print("   Traditional search would miss many of these matches.")


def search_mode(query: str):
    """Quick search mode."""
    engine = EmbeddingEngine()
    engine.add_documents(SAMPLE_DOCS)
    
    results = engine.search(query, top_k=5)
    
    print(f"\n🔍 Search: '{query}'")
    print("-" * 70)
    
    for i, (doc_id, score, text) in enumerate(results, 1):
        print(f"\n{i}. [Score: {score:.3f}]")
        print(f"   {text}")


def explain_mode(text: str):
    """Explain what an embedding looks like."""
    engine = EmbeddingEngine()
    engine.add_documents(SAMPLE_DOCS)
    engine.explain_embedding(text, top_n=10)


def similar_mode(doc_id: int):
    """Find similar documents."""
    engine = EmbeddingEngine()
    engine.add_documents(SAMPLE_DOCS)
    
    if doc_id >= len(SAMPLE_DOCS):
        print(f"Error: Document ID {doc_id} not found")
        print(f"Valid range: 0-{len(SAMPLE_DOCS)-1}")
        return
    
    print(f"\nDocument #{doc_id}:")
    print(f"'{SAMPLE_DOCS[doc_id]}'")
    
    results = engine.find_similar(doc_id, top_k=3)
    
    print(f"\n🔗 Similar documents:")
    print("-" * 70)
    
    for i, (idx, score, text) in enumerate(results, 1):
        print(f"\n{i}. [Score: {score:.3f}]")
        print(f"   {text}")


def compare_mode(text1: str, text2: str):
    """Compare similarity between two texts."""
    engine = EmbeddingEngine()
    engine.add_documents([text1, text2])  # Need corpus for TF-IDF
    
    similarity = engine.compute_similarity(text1, text2)
    
    print("\n" + "=" * 70)
    print("SIMILARITY COMPARISON")
    print("=" * 70)
    print(f"\nText 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"\nSimilarity Score: {similarity:.3f}")
    
    if similarity > 0.8:
        print("\n✅ Very similar! (Almost the same meaning)")
    elif similarity > 0.5:
        print("\n🔶 Somewhat similar (Related topics)")
    elif similarity > 0.2:
        print("\n🔷 Slightly similar (Some overlap)")
    else:
        print("\n❌ Not similar (Different topics)")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("""
╔══════════════════════════════════════════════════════════╗
║        SEMANTIC SEARCH ENGINE                            ║
║        Learn Embeddings by Searching!                    ║
╚══════════════════════════════════════════════════════════╝

Usage:
  python main.py demo                    # Interactive demo
  python main.py search "your query"     # Search documents
  python main.py similar <doc_id>        # Find similar docs
  python main.py explain "some text"     # Show embedding
  python main.py compare "text1" "text2" # Compare similarity
  python main.py concept                 # Learn embedding concepts

Examples:
  python main.py demo
  python main.py search "climate change"
  python main.py similar 0
  python main.py explain "machine learning"
  python main.py compare "car" "automobile"

What are Embeddings?
  Embeddings convert text → numbers (vectors)
  Similar text → Similar numbers
  This enables semantic (meaning-based) search!

Why This Matters:
  • Google Search uses embeddings
  • ChatGPT uses embeddings
  • All modern AI uses embeddings
  
  This project shows you HOW they work! 🚀
        """)
        return
    
    command = sys.argv[1].lower()
    
    if command == 'demo':
        demo_mode()
    
    elif command == 'search':
        if len(sys.argv) < 3:
            print("Usage: python main.py search \"your query\"")
        else:
            query = ' '.join(sys.argv[2:])
            search_mode(query)
    
    elif command == 'similar':
        if len(sys.argv) < 3:
            print("Usage: python main.py similar <doc_id>")
        else:
            try:
                doc_id = int(sys.argv[2])
                similar_mode(doc_id)
            except ValueError:
                print("Error: doc_id must be a number")
    
    elif command == 'explain':
        if len(sys.argv) < 3:
            print("Usage: python main.py explain \"some text\"")
        else:
            text = ' '.join(sys.argv[2:])
            explain_mode(text)
    
    elif command == 'compare':
        if len(sys.argv) < 4:
            print("Usage: python main.py compare \"text1\" \"text2\"")
        else:
            # Find the two quoted strings
            full_text = ' '.join(sys.argv[2:])
            parts = full_text.split('" "')
            if len(parts) >= 2:
                text1 = parts[0].strip('"')
                text2 = parts[1].strip('"')
                compare_mode(text1, text2)
            else:
                print("Please provide two quoted strings")
    
    elif command == 'concept':
        demonstrate_embeddings()
    
    else:
        print(f"Unknown command: {command}")
        print("Try: python main.py (with no arguments) for help")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
        sys.exit(0)
