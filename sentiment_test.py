from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", revision="714eb0f")

# Test with sample text
texts = [
    "This movie was amazing!",
    "I hated the ending of the book.",
    "The weather is okay today."
]
for text in texts:
    result = sentiment_pipeline(text)  # Use sentiment_pipeline instead of classifier
    print(f"Text: {text}")
    print(f"Sentiment: {result[0]['label']}, Confidence: {result[0]['score']:.3f}\n")