import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Configuration ---
st.set_page_config(page_title="Dry Fruit Recommendation System", layout="centered")

# ==========================================
# --- DATA INITIALIZATION & CLEANING --------
# ==========================================
@st.cache_data # Caches the data preparation so it doesn't run on every button click
def load_and_clean_data():
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
            'Sweet whole almonds glazed with honey and roasted until crispy.', # Handled missing value natively
            'A classic party mix of lightly salted cashews, almonds, and crunchy pistachios.',
            'Gourmet cashews tossed in a tangy, hot African Peri Peri seasoning mix.'
        ]
    }
    df = pd.DataFrame(product_data)
    
    # Custom baseline cleaning regex
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        stopwords = {'a', 'about', 'and', 'are', 'for', 'in', 'is', 'it', 'of', 'on', 'the', 'to', 'with', 'or'}
        return " ".join([word for word in text.split() if word not in stopwords])

    df['clean_text'] = df['Description'].apply(clean_text)
    return df

df = load_and_clean_data()

# ==========================================
# --- VECTORIZATION & SIMILARITY CONTROLS ---
# ==========================================
# Compute TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])

# Compute global cosine similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)


# ==========================================
# --- STREAMLIT USER INTERFACE (Task 6) -----
# ==========================================
st.title("🥜 FlavorMatch Recommendation System")
st.write("Select a product below, and our content-based engine will discover items with matching flavor profiles.")
st.markdown("---")

# 1. Dropdown to select item
selected_product = st.selectbox(
    "Choose a product you like:",
    options=df['Product_Name'].tolist()
)

# 2. Button to generate recommendations
if st.button("Generate Recommendations", type="primary"):
    
    # Recommendation logic lookup
    idx = df[df['Product_Name'] == selected_product].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    
    # Sort items by similarity score in descending order
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    st.subheader(f"Top matches for '{selected_product}':")
    
    # Display top 3 recommendations (skipping index 0 since that's the item itself)
    count = 0
    for score_pair in sorted_scores[1:4]:
        rec_idx = score_pair[0]
        rec_score = score_pair[1]
        
        # Check if there's any textual overlapping similarity (score > 0)
        if rec_score > 0:
            count += 1
            with st.container():
                st.markdown(f"### {count}. {df.loc[rec_idx, 'Product_Name']}")
                st.caption(f"Match Confidence Score: {rec_score * 100:.1f}%")
                st.write(df.loc[rec_idx, 'Description'])
                st.markdown("")
                
    if count == 0:
        st.info("No highly correlated alternative flavors found in the current small catalog layer.")