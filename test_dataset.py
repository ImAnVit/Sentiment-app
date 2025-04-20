import pandas as pd
from sentiment_analyzer import analyze_sentiment

# Load dataset
df = pd.read_csv("IMDb.csv")  # Adjust path if needed
sample_reviews = df["review"].head(5)  # First 5 reviews

# Analyze sentiment
for review in sample_reviews:
    result = analyze_sentiment(review[:512])  # Limit to 512 tokens (model constraint)
    print(f"Review: {review[:100]}...")  # Print first 100 chars
    print(f"Sentiment: {result['label']}, Confidence: {result['score']:.3f}\n")