import streamlit as st
from sentiment_analyzer import analyze_sentiment
import plotly.express as px
import pandas as pd

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 5px;
}
.stTextArea textarea {
    border: 2px solid #007bff;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

st.title("Sentiment Analysis Web App")
st.markdown("Analyze the sentiment of your text (e.g., movie reviews, tweets).")

# Sidebar with example texts
st.sidebar.header("Try Examples")
examples = [
    "This movie was absolutely fantastic!",
    "I'm so disappointed with this product.",
    "The weather is okay today."
]
for example in examples:
    if st.sidebar.button(example[:30] + "...", key=f"example_{examples.index(example)}"):
        st.session_state.user_input = example

# Main input area
# Two-column layout
col1, col2 = st.columns([2, 1])

with col1:
    # Text input
    user_input = st.text_area(
        "Input Text", 
        value=st.session_state.get("user_input", ""),
        placeholder="Type your text here...", 
        height=150, 
        key="user_input"
    )
    
    # Analyze button
    if st.button("Analyze Sentiment", key="analyze_btn"):
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

with col2:
    # Tips section
    st.write("**Tips**")
    st.write("- Enter up to 512 characters.")
    st.write("- Try movie reviews or tweets.")
    st.write("- Use clear, expressive text.")

# File upload
st.subheader("Batch Analysis")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="file_uploader")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "text" in df.columns:
        results = []
        for text in df["text"]:
            result = analyze_sentiment(str(text)[:512])
            results.append({
                "text": str(text)[:100],  # Truncate for display
                "label": result["label"],
                "score": result["score"]
            })
        df_results = pd.DataFrame(results)
        st.write("Batch Analysis Results")
        st.dataframe(df_results[["text", "label", "score"]])
        # Save results to history
        st.session_state.history.extend(results)
    else:
        st.error("CSV must have a 'text' column.")

# Display history
if st.session_state.history:
    if st.button("Clear History", key="clear_history_btn"):
        st.session_state.history = []
        st.rerun()
    st.subheader("Analysis History")
    for i, entry in enumerate(st.session_state.history):
        st.write(f"**Entry {i+1}**: {entry['text']}...")
        st.write(f"Sentiment: {entry['label']}, Confidence: {entry['score']:.3f}")

    # Create pie chart
    df = pd.DataFrame(st.session_state.history)
    sentiment_counts = df["label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    fig = px.pie(sentiment_counts, names="Sentiment", values="Count", title="Sentiment Distribution")
    st.plotly_chart(fig)
    
    # Create bar chart
    fig_bar = px.bar(df, x=df.index, y="score", color="label", title="Confidence Scores")
    st.plotly_chart(fig_bar)
