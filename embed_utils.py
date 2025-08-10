import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import logging

logger = logging.getLogger(__name__)

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_text_embedding(text):
    """Get embedding for a single text"""
    try:
        if isinstance(text, list) and len(text) > 0:
            text = text[0]  # Take first element if it's a list
        return model.encode(str(text))
    except Exception as e:
        logger.error(f"Error getting text embedding: {e}")
        return np.zeros(384)  # Return zero vector of appropriate dimension

def calculate_text_similarity(sentence, reference_sentence):
    """
    Calculate cosine similarity and euclidean distance between two sentences
    
    Args:
        sentence (str): The first sentence
        reference_sentence (str): The reference sentence to compare against
        
    Returns:
        tuple: (cosine_similarity_score, euclidean_distance)
    """
    try:
        # Handle empty or None inputs
        if not sentence or not reference_sentence:
            return np.array([[0.0]]), np.array([[1.0]])
        
        # Convert to strings if needed
        sentence = str(sentence).strip()
        reference_sentence = str(reference_sentence).strip()
        
        # Skip if either is empty after stripping
        if not sentence or not reference_sentence:
            return np.array([[0.0]]), np.array([[1.0]])
        
        # Get embeddings
        sentence_embedding = get_text_embedding(sentence)
        reference_embedding = get_text_embedding(reference_sentence)
        
        # Reshape for sklearn functions
        sentence_embedding = sentence_embedding.reshape(1, -1)
        reference_embedding = reference_embedding.reshape(1, -1)
        
        # Calculate similarities
        cos_sim = cosine_similarity(sentence_embedding, reference_embedding)
        eucl_dist = euclidean_distances(sentence_embedding, reference_embedding)
        
        return cos_sim, eucl_dist
        
    except Exception as e:
        logger.error(f"Error calculating text similarity: {e}")
        # Return default values in case of error
        return np.array([[0.0]]), np.array([[1.0]])

def batch_calculate_similarity(sentences, reference_sentences):
    """
    Calculate similarity for batches of sentences
    
    Args:
        sentences (list): List of sentences
        reference_sentences (list): List of reference sentences
        
    Returns:
        tuple: (list of cosine similarities, list of euclidean distances)
    """
    similarities = []
    distances = []
    
    for sent, ref in zip(sentences, reference_sentences):
        sim, dist = calculate_text_similarity(sent, ref)
        similarities.append(float(sim[0][0]))
        distances.append(float(dist[0][0]))
    
    return similarities, distances