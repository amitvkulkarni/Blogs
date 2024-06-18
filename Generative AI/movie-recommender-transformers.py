import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("imdb_top_1000.csv")

# Display the first few rows of the DataFrame
df.head()

# Select all columns except the first one for combination
columns_to_combine = df.iloc[:, 1:]

# Create a new column 'combined_text' by concatenating selected columns
df["combined_text_tmp"] = df[columns_to_combine.columns].apply(
    lambda row: " ".join(row.values.astype(str)), axis=1
)

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/paraphrase-MiniLM-L6-v2"
)
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")


def compute_embeddings(df, tokenizer, model):
    """
    Function to compute embeddings for the combined text in the DataFrame.
    Args:
    df : DataFrame
    tokenizer : Tokenizer
    model : Model

    Returns:
    embeddings : Embeddings for the combined text
    """
    # Combine relevant columns into one input text
    combined_text = df["combined_text_tmp"]

    # Tokenize the combined text
    inputs = tokenizer(
        combined_text.tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    # Compute embeddings using the model
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings


# Compute embeddings for the relevant columns
embeddings = compute_embeddings(df, tokenizer, model)

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Print the similarity matrix
print(similarity_matrix)


def recommend_movies(
    user_input, df, embeddings, similarity_matrix, tokenizer, model, top_k=10
):
    """
    Function to recommend movies based on user input.
    Args:
    user_input : str
    df : DataFrame
    embeddings : Embeddings
    similarity_matrix : Similarity Matrix
    tokenizer : Tokenizer
    model : Model
    top_k : int

    Returns:
    recommended_movies : DataFrame of recommended movies
    """
    # Tokenize the user input
    input_tokens = tokenizer(
        user_input, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    # Compute embedding for user input
    with torch.no_grad():
        input_embedding = model(**input_tokens).last_hidden_state.mean(dim=1)

    # Compute cosine similarity between user input and all movie overviews
    similarities = cosine_similarity(input_embedding, embeddings)

    # Get indices of top_k most similar movies
    similar_movie_indices = similarities[0].argsort()[-top_k:][::-1]

    # Get top_k similar movies from the dataframe
    recommended_movies = df.iloc[similar_movie_indices]

    # Add a column for similarity score (optional)
    recommended_movies["Similarity Score"] = similarities[0][similar_movie_indices]

    return recommended_movies


# User input for movie recommendation
user_input = "Super heros"

# Get recommendations
recommendations = recommend_movies(
    user_input, df, embeddings, similarity_matrix, tokenizer, model
)

# Display recommendations in a DataFrame
print("Recommendations based on input:")
recommendations[["Series_Title", "Genre", "Overview", "IMDB_Rating"]]
