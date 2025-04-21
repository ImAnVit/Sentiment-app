import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from sentiment_analyzer import analyze_sentiment

# Streamlit app
st.title("🎬 Sentiment Analysis Web App")
st.markdown("Analyze the sentiment of your text (e.g., movie reviews, tweets).")

# Text input
user_input = st.text_area("Input Text", placeholder="Type your text here...", height=150)

# Analyze button
if st.button("Analyze Sentiment"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            result = analyze_sentiment(user_input[:512])  # Limit to 512 tokens
        st.subheader("Results")
        if result["label"] == "POSITIVE":
            st.success(f"😊 Sentiment: {result['label']}")
        else:
            st.error(f"😔 Sentiment: {result['label']}")
        st.write(f"**Confidence**: {result['score']:.3f}")
        st.progress(result["score"])  # Visual confidence bar
    else:
        st.warning("Please enter some text.")

# Add example texts
st.sidebar.header("Try Examples")
examples = [
    "This movie was absolutely fantastic!",
    "I’m so disappointed with this product.",
    "The weather is okay today."
]
for example in examples:
    if st.sidebar.button(example[:30] + "..."):
        st.text_area("Input Text", example)