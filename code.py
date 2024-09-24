import os
import math
from collections import defaultdict, Counter
import re

# Function to read text files from a specified directory
def load_files_from_directory(directory_path):
    docs = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):  # Process only text files
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                docs[filename] = content  # Use filename as Document ID
    return docs

# Function to split and preprocess text
def preprocess_text(text):
    # Convert to lowercase, remove non-word characters, and split into words
    return re.findall(r'\b\w+\b', text.lower())

# Step 1: Create the Inverted Index
def create_inverted_index(docs):
    inv_index = defaultdict(list)
    lengths = {}  # Store lengths of documents for normalization
    num_docs = len(docs)

    for doc_id, content in docs.items():
        tokens = preprocess_text(content)
        term_counts = Counter(tokens)
        length = 0
        for term, count in term_counts.items():
            tf = 1 + math.log10(count)  # Log-scaled term frequency
            length += tf ** 2
            inv_index[term].append((doc_id, count))
        lengths[doc_id] = math.sqrt(length)  # Calculate document length for cosine normalization

    return inv_index, lengths, num_docs

# Step 2: Calculate TF-IDF scores for the query terms
def calculate_query_weights(query, inv_index, num_docs):
    query_tokens = preprocess_text(query)
    term_counts = Counter(query_tokens)
    weights = {}
    
    for term, count in term_counts.items():
        if term in inv_index:
            doc_freq = len(inv_index[term])
            idf = math.log10(num_docs / doc_freq)  # Inverse Document Frequency
            tf = 1 + math.log10(count)  # Log-scaled term frequency
            weights[term] = tf * idf
        else:
            weights[term] = 0
    
    return weights

# Step 3: Score documents based on Cosine Similarity
def score_documents(query_weights, inv_index, lengths):
    scores = defaultdict(float)

    for term, weight in query_weights.items():
        if term in inv_index:
            postings = inv_index[term]
            for doc_id, count in postings:
                doc_weight = 1 + math.log10(count)  # Log-scaled term frequency for document
                scores[doc_id] += weight * doc_weight
    
    # Normalize scores by document length
    for doc_id in scores:
        scores[doc_id] /= lengths[doc_id]
    
    # Sort scores
    sorted_docs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

    # Prepare output as tuples (filename, similarity_score)
    results = [(doc, round(score, 17)) for doc, score in sorted_docs]

    return results[:10]  # Return top 10 documents


if __name__ == "__main__":
    directory_path = r"C:\Users\Sharan PY\Desktop\7th Semester\IR\assignment 2\corpus"  # Update with your folder path
    docs = load_files_from_directory(directory_path)
    
    # Create the index and calculate document lengths
    inv_index, lengths, num_docs = create_inverted_index(docs)
    
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        query_weights = calculate_query_weights(user_query, inv_index, num_docs)
        ranked_documents = score_documents(query_weights, inv_index, lengths)
        print("Top 10 Relevant Documents:", ranked_documents)
