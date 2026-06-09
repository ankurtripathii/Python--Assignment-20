import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# --- DATA SETUP & PREPROCESSING (PART 1) ---
# ==========================================
product_data = {
    'Product_ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'Product_Name': [
        'Salty Pepper Cashews', 'Spicy Chili Almonds', 'Classic Roasted Walnuts', 
        'Black Pepper Pistachios', 'Organic Raisins', 'Premium Dates', 
        'Hot & Spicy Mix', 'Honey Glazed Almonds', 'Salted Mixed Nuts', 'Peri Peri Cashews'
    ],
    'Description': [
        'Premium crunchy cashews coated with premium sea salt and freshly crushed black pepper!',
        'Crispy California almonds dusted with a fiery red chili and cayenne pepper spice blend.',
        'Rich, earthy walnuts lightly roasted to perfection. 100% natural and raw flavor.',
        'Toasted pistachios in shell, seasoned heavily with dynamic black pepper & salt flakes.',
        'Sweet and chewy sun-dried organic raisins. Perfect for evening snacking or baking.',
        'Soft, sweet premium dates sourced directly from farms. Rich in fiber and natural sugars.',
        'A burning hot and spicy blend of peanuts, cashews, and almonds for spice lovers!!!',
        np.nan, # Missing values handling test
        'A classic party mix of lightly salted cashews, almonds, and crunchy pistachios.',
        'Gourmet cashews tossed in a tangy, hot African Peri Peri seasoning mix.'
    ]
}
df = pd.DataFrame(product_data)

# Baseline Text Cleaner
def clean_recommendation_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    stopwords = {'a', 'about', 'and', 'are', 'for', 'in', 'is', 'it', 'of', 'on', 'the', 'to', 'with', 'or', 'and'}
    return " ".join([word for word in text.split() if word not in stopwords])

df['clean_text'] = df['Description'].apply(clean_recommendation_text)


# ==========================================
# --- PART 2: TEXT VECTORIZATION (TASK 3) ---
# ==========================================
print("--- PART 2: Text Vectorization ---")
print("--- Task 3: Vectorization using TF-IDF ---")

# 1. Use TfidfVectorizer() to convert text into vectors.
# 2. Set reasonable parameters: 
#    - max_features: caps vocabulary size to the top N most important tokens
#    - ngram_range: (1, 2) extracts both single words and two-word phrases (e.g., "black pepper")
tfidf_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))

# Fit and transform the processed column
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])

# 3. Display shape of TF-IDF matrix
print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape} -> ({tfidf_matrix.shape[0]} Products, {tfidf_matrix.shape[1]} Unique Features/N-grams)\n")


# ==========================================
# --- SIMILARITY COMPUTATION (TASK 4) -------
# ==========================================
print("-" * 50)
print("--- Task 4: Similarity Computation ---")

# 1 & 2. Compute and store cosine similarity between all items
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Print a preview slice of the matrix (First 3 products across first 3 products)
print("Cosine Similarity Matrix Preview (First 3x3 Block):")
print(np.round(similarity_matrix[:3, :3], 4))


# ==========================================
# --- PART 3: RECOMMENDATION LOGIC (TASK 5) -
# ==========================================
print("\n" + "="*50)
print("--- PART 3: Recommendation Logic ---")
print("--- Task 5: Build Recommendation Function ---")
print("="*50)

def recommend(item_name, top_n=3):
    # Normalize input and product names for case-insensitive lookup
    item_name_lower = item_name.strip().lower()
    product_names_lower = df['Product_Name'].str.lower().str.strip()
    
    # Check if the product exists in our database index
    if item_name_lower not in product_names_lower.values:
        return f"Error: '{item_name}' was not found in the product database."
    
    # 1. Find index of the selected item
    idx = df[product_names_lower == item_name_lower].index[0]
    
    # 2. Compute / extract similarity scores for this specific index
    # Enumerate helps track the original product index position alongside its score
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    
    # 3. Sort item arrays in descending order based on similarity scores
    # lambda x: x[1] targets the score value for sorting priority
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Extract indices of top recommendations
    # [1:top_n+1] slices the list to bypass index 0, because the most similar item to a product is always itself (score 1.0)
    top_indices = [score_pair[0] for score_pair in sorted_scores[1:top_n+1]]
    
    # Return top recommendations as a clean DataFrame view
    return df[['Product_Name', 'Description']].iloc[top_indices].reset_index(drop=True)

# --- Test with at least 3 different items ---
test_items = ['Salty Pepper Cashews', 'Spicy Chili Almonds', 'Hot & Spicy Mix']

for item in test_items:
    print(f"\n[Recommendations for: '{item}']")
    recommendations = recommend(item, top_n=2)
    print(recommendations)