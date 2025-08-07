import os
from mistralai import Mistral
from sklearn.metrics.pairwise import cosine_similarity , euclidean_distances
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-embed"

# client = Mistral(api_key=api_key)


# def get_text_embedding(inputs):
#     embeddings_batch_response = client.embeddings.create(
#         model=model,
#         inputs=inputs
#     )
#     return embeddings_batch_response.data[0].embedding

# def calculate_text_similarity(sentence, reference_sentence):
#     embeddings = get_text_embedding([sentence])
#     reference_embedding = get_text_embedding([reference_sentence])

#     return cosine_similarity([embeddings], [reference_embedding]) , euclidean_distances([embeddings], [reference_embedding])
#     # return embeddings, reference_embedding

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_text_embedding(inputs):
    return model.encode(inputs[0])

def calculate_text_similarity(sentence, reference_sentence):
    embeddings = get_text_embedding([sentence])
    reference_embedding = get_text_embedding([reference_sentence])

    return cosine_similarity([embeddings], [reference_embedding]), euclidean_distances([embeddings], [reference_embedding])