import pandas as pd
import numpy as np
import re

print("--- PART 1: Data Preprocessing ---")

# --- Step 1: Create/Load the Dataset ---
# Creating a custom product catalog dataset to simulate an e-commerce database
product_data = {
    'Product_ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'Product_Name': [
        'Salty Pepper Cashews', 'Spicy Chili Almonds', 'Classic Roasted Walnuts', 
        'Black Pepper Pistachios', 'Organic Raisins', 'Premium Dates', 
        'Hot & Spicy Mix', 'Honey Glazed Almonds', 'Salted Mixed Nuts', 'Peri Peri Cashews'
    ],
    'Category': ['Cashew', 'Almond', 'Walnut', 'Pistachio', 'Raisin', 'Dates', 'Mix', 'Almond', 'Mix', 'Cashew'],
    'Description': [
        'Premium crunchy cashews coated with premium sea salt and freshly crushed black pepper!',
        'Crispy California almonds dusted with a fiery red chili and cayenne pepper spice blend.',
        'Rich, earthy walnuts lightly roasted to perfection. 100% natural and raw flavor.',
        'Toasted pistachios in shell, seasoned heavily with dynamic black pepper & salt flakes.',
        'Sweet and chewy sun-dried organic raisins. Perfect for evening snacking or baking.',
        'Soft, sweet premium dates sourced directly from farms. Rich in fiber and natural sugars.',
        'A burning hot and spicy blend of peanuts, cashews, and almonds for spice lovers!!!',
        np.nan, # Intentionally leaving a missing value to test step 2.4 handling
        'A classic party mix of lightly salted cashews, almonds, and crunchy pistachios.',
        'Gourmet cashews tossed in a tangy, hot African Peri Peri seasoning mix.'
    ]
}

# 1. Load the dataset using Pandas
df = pd.DataFrame(product_data)


# --- Task 1: Load & Understand Dataset ---
print("\n" + "="*50)
print("--- Task 1: Load & Understand Dataset ---")
print("="*50)

# 2. Print essential details
print("\n[A] Dataset Shape (Rows, Columns):")
print(df.shape)

print("\n[B] Column Names:")
print(df.columns.tolist())

print("\n[C] First 5 Rows:")
print(df.head())

print("\n[D] Other Essential Details (Info summary):")
print(df.info())

# 3. Identify text column(s) used for recommendations
print("\n[E] Recommendation Feature Identification:")
print("The primary text column identified for building content-based recommendations is: 'Description'.")
print("This column contains the critical semantic keywords (e.g., 'salty', 'spicy', 'pepper') that reflect product attributes.")


# --- Task 2: Text Preprocessing for Recommendation ---
print("\n" + "="*50)
print("--- Task 2: Text Preprocessing for Recommendation ---")
print("="*50)

# Simple custom list of English stopwords to minimize external dependency crashes
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'arent', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cant', 'cannot',
    'could', 'did', 'didnt', 'do', 'does', 'doesnt', 'doing', 'dont', 'down', 'during', 'each', 'few', 'for', 'from',
    'further', 'had', 'hadnt', 'has', 'hasnt', 'have', 'havent', 'having', 'he', 'her', 'here', 'hers', 'herself',
    'him', 'himself', 'his', 'how', 'if', 'in', 'into', 'is', 'isnt', 'it', 'its', 'itself', 'just', 'me', 'more',
    'most', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their',
    'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too',
    'under', 'until', 'up', 'very', 'was', 'wasnt', 'we', 'were', 'werent', 'what', 'when', 'where', 'which', 'while',
    'who', 'whom', 'why', 'with', 'wont', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves'
}

def clean_recommendation_text(text):
    # 4. Handle missing values: If the row is NaN, safely return an empty string
    if pd.isna(text):
        return ""
        
    # 1. Convert text to lowercase
    text = str(text).lower()
    
    # 2. Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', ' ', text) # Replaces non-word characters with a single space
    
    # 3. Remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in STOPWORDS]
    
    # Re-join filtered tokens back into a single cohesive string, stripping out extra spaces
    return " ".join(filtered_words)

# 5. Store cleaned text in a new column: clean_text
df['clean_text'] = df['Description'].apply(clean_recommendation_text)

print("\nFinal Preprocessed Output (Original Description vs Cleaned Text):")
# Displaying the Product Name, Original Description, and our clean_text side-by-side
print(df[['Product_Name', 'Description', 'clean_text']].head(8))