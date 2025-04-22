import streamlit as st
from sentiment_analyzer import analyze_sentiment

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Sentiment Analysis Web App")
st.markdown("Analyze the sentiment of your text (e.g., movie reviews, tweets).")

# Text input
user_input = st.text_area("Input Text", placeholder="Type your text here...", height=150)

# Analyze button
if st.button("Analyze Sentiment"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            result = analyze_sentiment(user_input[:512])
        st.subheader("Results")
        if result["label"] == "POSITIVE":
            st.success(f"😊 Sentiment: {result['label']}")
        else:
            st.error(f"😔 Sentiment: {result['label']}")
        st.write(f"**Confidence**: {result['score']:.3f}")
        st.progress(result["score"])
        # Store result in history
        st.session_state.history.append({
            "text": user_input[:100],  # Truncate for display
            "label": result["label"],
            "score": result["score"]
        })
    else:
        st.warning("Please enter some text.")

# Display history
if st.session_state.history:
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
    st.subheader("Analysis History")
    for i, entry in enumerate(st.session_state.history):
        st.write(f"**Entry {i+1}**: {entry['text']}...")
        st.write(f"Sentiment: {entry['label']}, Confidence: {entry['score']:.3f}")