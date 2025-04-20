from transformers import pipeline

# Create the pipeline ONCE
classifier = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """
    Analyze sentiment of input text.
    Returns: Dictionary with label and confidence score.
    """
    result = classifier(text)
    return {
        "text": text,
        "label": result[0]["label"],
        "score": result[0]["score"]
    }

# Test the function
if __name__ == "__main__":
    sample_texts = [
        "This is the best day ever!",
        "I'm so disappointed with this product.",
        "It's an average experience."
    ]
    for text in sample_texts:
        result = analyze_sentiment(text)
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['label']}, Confidence: {result['score']:.3f}\n")