# Semantic Search Engine 🔍

A **simple but powerful** document search engine that uses **embedding principles** to find semantically similar content. Instead of just matching keywords, it understands **meaning**!

## 🧠 What Are Embeddings?

### The Simple Explanation
**Embeddings** convert text into numbers (vectors) that capture **meaning**:
- Similar meanings → Similar numbers
- Different meanings → Different numbers

### Example
```
"dog" → [0.8, 0.2, 0.1, 0.9, ...]
"cat" → [0.7, 0.3, 0.2, 0.8, ...]  # Similar to dog!
"car" → [0.1, 0.8, 0.9, 0.2, ...]  # Very different!
```

Notice "dog" and "cat" have similar vectors because they're both animals!

### Why Embeddings Are Powerful

**Traditional Keyword Search:**
```
Query: "automobile"
Document: "I bought a car"
Result: ❌ NO MATCH (different words!)
```

**Embedding-Based Search:**
```
Query: "automobile" → [0.1, 0.8, 0.9, 0.2, ...]
Document: "I bought a car" → [0.1, 0.8, 0.9, 0.2, ...]
Result: ✅ MATCH! (same meaning!)
```

## ✨ Features

### 🎯 Semantic Understanding
- Finds documents by **meaning**, not just keywords
- Understands synonyms automatically
- Handles typos and variations

### 🚀 Simple Implementation
- Only ~200 lines of code
- Uses TF-IDF for basic embeddings
- No neural networks needed
- Pure Python + scikit-learn

### 📊 Practical Applications
- Search through documents
- Find similar articles
- Cluster related content
- Detect duplicates

### 🎓 Educational
- Learn embedding principles
- Understand vector similarity
- See dimensionality in action

## 🚀 Quick Start

```bash
# Setup
cd semantic-search
pip install -r requirements.txt

# Run demo
python main.py demo

# Search documents
python main.py search "climate change effects"

# Add your own documents
python main.py add "Your document text here"

# Find similar documents
python main.py similar 0  # Find docs similar to document #0
```

## 📖 How Embeddings Work

### Step 1: Text to Numbers
```python
# Text
text = "The cat sat on the mat"

# Embedding (vector)
embedding = [0.2, 0.8, 0.1, 0.5, 0.3, ...]
# Each number represents some "meaning dimension"
```

### Step 2: Measuring Similarity
```python
# Two documents
doc1 = "I love dogs"
doc2 = "I adore cats"
doc3 = "The sky is blue"

# Embeddings
emb1 = [0.8, 0.2, 0.1]
emb2 = [0.7, 0.3, 0.2]  # Similar to doc1!
emb3 = [0.1, 0.1, 0.9]  # Very different!

# Similarity scores
similarity(doc1, doc2) = 0.95  # Very similar!
similarity(doc1, doc3) = 0.23  # Not similar
```

### Step 3: Finding Matches
```python
query = "puppy"
query_embedding = [0.75, 0.25, 0.1]

# Compare with all documents
best_match = doc1  # Closest embedding to query!
```

## 🎯 Embedding Techniques Used

### 1. TF-IDF (Term Frequency - Inverse Document Frequency)
```
Basic idea:
- Common words (the, is, a) → Low importance
- Rare words (quantum, embryonic) → High importance
- Word frequency in doc → Relevance

Result: Each document becomes a vector of word importances
```

### 2. Cosine Similarity
```
Measures angle between vectors:
- Cosine = 1.0 → Identical direction (very similar)
- Cosine = 0.5 → Different but related
- Cosine = 0.0 → Completely different
```

### Visual Explanation
```
2D Space Visualization:

        dog •
            |
            |  cat •
            | /
            |/___________
           /|        car •
          / |
         /  |
    puppy • |
```

"dog", "cat", "puppy" are close together (similar embeddings)
"car" is far away (different embedding)

## 📊 Example Use Cases

### Use Case 1: Document Search
```python
# Add documents
engine.add("Paris is the capital of France")
engine.add("London is in England")
engine.add("I love pizza and pasta")

# Search
results = engine.search("French capital city")

# Returns: "Paris is the capital of France"
# Even though query doesn't contain exact words!
```

### Use Case 2: Finding Similar Content
```python
# Find documents similar to #0
similar = engine.find_similar(doc_id=0, top_k=3)

# Returns documents with similar meaning
```

### Use Case 3: Duplicate Detection
```python
# These are duplicates (high similarity)
doc1 = "Apple releases new iPhone"
doc2 = "Apple launches latest iPhone model"

similarity = engine.compute_similarity(doc1, doc2)
# Result: 0.92 (very similar!)
```

## 🎓 Understanding the Code

### Core Concept: Vector Space
```python
# Each document is a point in high-dimensional space

Document 1: [0.2, 0.8, 0.1, 0.5]
Document 2: [0.3, 0.7, 0.2, 0.4]  # Close to Doc 1
Document 3: [0.9, 0.1, 0.8, 0.1]  # Far from Doc 1

# Distance = Similarity
# Close vectors = Similar meaning
```

### The Magic of Dimensions
```
Think of each dimension as a "concept":
- Dimension 0: "Animal-ness"
- Dimension 1: "Technology-ness"
- Dimension 2: "Food-ness"
- ... (hundreds more)

Document about cats: [0.9, 0.1, 0.0, ...]
Document about phones: [0.0, 0.9, 0.1, ...]
Document about pizza: [0.0, 0.0, 0.9, ...]
```

## 🛠️ Technical Details

### TF-IDF Explained
```python
# TF (Term Frequency): How often word appears in document
tf = count(word, document) / total_words(document)

# IDF (Inverse Document Frequency): How rare the word is
idf = log(total_documents / documents_containing(word))

# TF-IDF: Combines both
tfidf = tf * idf

# Result: Common words get low scores, rare words get high scores
```

### Why This Works
```
Document: "The cat sat on the mat"

TF-IDF scores:
"the" → 0.01  (very common word, low score)
"cat" → 0.78  (less common, high score)
"sat" → 0.65  (medium commonality)

The vector [0.01, 0.78, 0.65, ...] represents the document!
```

## 📈 Comparison with Keyword Search

### Keyword Search
```python
Query: "automobile accident"
Doc 1: "car crash on highway" → ❌ No match
Doc 2: "automobile accident report" → ✅ Match

Problem: Misses synonyms!
```

### Embedding Search
```python
Query: "automobile accident" → [0.7, 0.8, 0.1]
Doc 1: "car crash" → [0.72, 0.79, 0.12] → ✅ Match! (0.98 similarity)
Doc 2: "automobile accident" → [0.7, 0.8, 0.1] → ✅ Match! (1.0 similarity)

Success: Finds both!
```

## 🎯 Project Structure

```
semantic-search/
├── main.py                 # CLI interface
├── requirements.txt        # Just scikit-learn & numpy!
├── src/
│   ├── embeddings.py      # Embedding engine
│   ├── search.py          # Search functionality
│   └── utils.py           # Helper functions
├── data/
│   └── documents.json     # Sample documents
└── examples/
    └── demo.py            # Example usage
```

## 💡 Example Sessions

### Session 1: Basic Search
```bash
$ python main.py demo

Loading sample documents...
✓ Loaded 10 documents

Enter search query: global warming

Top 3 Results:
1. [Score: 0.89] Climate change is affecting polar ice caps...
2. [Score: 0.76] Rising temperatures impact ecosystems worldwide...
3. [Score: 0.62] Environmental policies target carbon emissions...

Notice: Found "climate change" even though query was "global warming"!
```

### Session 2: Finding Similarities
```bash
$ python main.py similar 0

Document #0:
"Artificial intelligence is transforming technology"

Similar documents:
1. [Score: 0.91] Machine learning advances computing capabilities
2. [Score: 0.85] AI revolutionizes software development
3. [Score: 0.73] Neural networks improve automation
```

### Session 3: Adding Documents
```bash
$ python main.py add "Quantum computing breakthrough announced"

✓ Document added (ID: 11)
✓ Embeddings updated

Similar existing documents:
1. [Score: 0.82] Scientists achieve quantum supremacy
2. [Score: 0.71] Computing technology advances rapidly
```

## 🧪 Understanding Embeddings Visually

### 1D Example (Simplified)
```
"dog" → 0.8
"cat" → 0.7
"car" → 0.1

Distance(dog, cat) = |0.8 - 0.7| = 0.1 (close!)
Distance(dog, car) = |0.8 - 0.1| = 0.7 (far!)
```

### 2D Example (Still Simple)
```
         Technology
              ↑
              |
    "phone" • |
              |
              |
    "laptop"• |
              |
━━━━━━━━━━━━━━•━━━━━━━━━→ Animals
              | •"cat"
              |  •"dog"
              |
```

### Real Example (300+ Dimensions!)
Each document has 300+ numbers representing different aspects of meaning.

## 🎓 Learning Exercises

### Exercise 1: Add Documents
```python
from src.embeddings import EmbeddingEngine

engine = EmbeddingEngine()
engine.add_document("Your text here")
engine.add_document("Another document")

# See how embeddings change
engine.show_embeddings()
```

### Exercise 2: Test Similarity
```python
# Which is more similar?
doc1 = "I love programming in Python"
doc2 = "Python is a great programming language"
doc3 = "I enjoy eating pizza"

# Predict: doc1 and doc2 should be very similar
# doc3 should be different
```

### Exercise 3: Synonym Search
```python
# Add documents with different words, same meaning
engine.add_document("The automobile is fast")
engine.add_document("The car is quick")

# Search with yet another synonym
results = engine.search("speedy vehicle")

# Should find both documents!
```

## 🔬 Advanced Concepts

### Why 300+ Dimensions?
```
More dimensions = More nuanced meaning

10 dimensions: Basic concepts (animal, technology, food)
100 dimensions: More detailed (pets, computers, Italian food)
300+ dimensions: Very specific nuances
```

### Dimensionality Reduction
```python
# Can visualize by reducing to 2D
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Now can plot on a graph!
```

## 🚀 Extending the Project

### Level 1: Basic Improvements
- Add more documents
- Tune similarity threshold
- Implement filters (date, category)

### Level 2: Better Embeddings
- Use Word2Vec
- Try GloVe embeddings
- Implement Doc2Vec

### Level 3: Neural Embeddings
- Use Sentence-BERT
- Try OpenAI embeddings
- Implement BERT

### Level 4: Production Features
- Add database storage
- Implement caching
- Build web API
- Add real-time updates

## 📚 Real-World Applications

### 1. Search Engines
- Google: Uses embeddings to understand query intent
- Find pages that MEAN what you search, not just match words

### 2. Recommendation Systems
- Netflix: Find similar movies
- Spotify: Recommend similar songs
- YouTube: Suggest related videos

### 3. Customer Support
- Find similar support tickets
- Suggest solutions based on past issues
- Route to right department

### 4. Content Moderation
- Detect duplicate posts
- Find similar spam
- Identify plagiarism

### 5. E-commerce
- "Customers also viewed..."
- Search product descriptions
- Find similar items

## 💻 Code Examples

### Example 1: Basic Usage
```python
from src.embeddings import EmbeddingEngine

# Create engine
engine = EmbeddingEngine()

# Add documents
engine.add_document("I love machine learning")
engine.add_document("Deep learning is fascinating")
engine.add_document("I enjoy cooking pasta")

# Search
results = engine.search("artificial intelligence")
# Returns documents about ML/DL, not cooking!
```

### Example 2: Similarity Matrix
```python
# Get similarity between all documents
matrix = engine.get_similarity_matrix()

# matrix[i][j] = similarity between doc i and doc j
print(f"Doc 0 vs Doc 1: {matrix[0][1]}")  # High (both about ML)
print(f"Doc 0 vs Doc 2: {matrix[0][2]}")  # Low (different topics)
```

### Example 3: Clustering
```python
# Find clusters of similar documents
clusters = engine.cluster_documents(n_clusters=3)

# All ML docs in one cluster
# All cooking docs in another cluster
```

## 🎯 Key Takeaways

1. **Embeddings = Text → Numbers**
   - Convert text to vectors that capture meaning

2. **Similar Meanings → Similar Vectors**
   - The magic of semantic search!

3. **Cosine Similarity**
   - Measure how "close" vectors are

4. **TF-IDF**
   - Simple but effective embedding method

5. **High Dimensions**
   - More dimensions = more nuanced understanding

## 🔮 Future Enhancements

Potential improvements:
- [ ] Word2Vec embeddings
- [ ] Neural embeddings (BERT)
- [ ] Multi-language support
- [ ] Image embeddings
- [ ] Audio embeddings
- [ ] Hybrid search (keywords + embeddings)

## 📝 Requirements

- Python 3.8+
- scikit-learn (for TF-IDF)
- numpy (for vector operations)

That's it! No huge models, no GPUs needed.

---

**Understand embeddings by building! 🚀**

*This project shows the core principle: text → vectors → similarity. Everything from Google Search to ChatGPT uses these same principles, just more sophisticated!*

## 🎓 Learn More

- **What are embeddings?** → Numbers that capture meaning
- **Why vectors?** → Easy to measure similarity with math
- **How does search work?** → Find closest vector to query
- **What's TF-IDF?** → Weight words by importance
- **Why cosine similarity?** → Best for high-dimensional vectors

**Start exploring semantic search today!** 🔍✨
