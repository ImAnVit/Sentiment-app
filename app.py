import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import torch
import traceback

# Set page title and configuration
st.set_page_config(page_title="Movie Sentiment & Chat App", layout="wide")

# Initialize sentiment analysis with VADER
try:
    # Create VADER sentiment analyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    # Function to analyze sentiment
    def analyze_sentiment(text):
        scores = sentiment_analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            return {"label": "POSITIVE", "score": (compound + 1) / 2}  # Scale to 0-1
        elif compound <= -0.05:
            return {"label": "NEGATIVE", "score": (1 - compound) / 2}  # Scale to 0-1
        else:
            return {"label": "NEUTRAL", "score": 0.5}
    
    st.success("✅ Sentiment analysis model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading sentiment analyzer: {str(e)}")
    st.stop()

# Initialize DialoGPT (in a try-except block to handle potential errors)
dialogpt_available = False
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Initialize with device_map="auto" and torch_dtype to handle meta tensors
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/DialoGPT-small", 
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32
    )
    
    # Test if model works
    test_input = tokenizer.encode("Hello" + tokenizer.eos_token, return_tensors="pt")
    model.generate(test_input, max_length=10, pad_token_id=tokenizer.eos_token_id)
    
    dialogpt_available = True
    st.success("✅ DialoGPT chatbot loaded successfully!")
except Exception as e:
    dialogpt_error = str(e)
    st.warning(f"⚠️ DialoGPT chatbot could not be loaded. Chat functionality will be limited.")

# Initialize session states
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🎬 Movie Sentiment & Chat App")
tabs = st.tabs(["Text Analysis", "Batch Analysis", "Movie Chat"])

# Text Analysis Tab
with tabs[0]:
    st.markdown("### Analyze the sentiment of your text")
    user_input = st.text_area("Input Text", placeholder="Type your text here...", height=150)
    
    if st.button("Analyze Sentiment", key="analyze_single"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                try:
                    result = analyze_sentiment(user_input)
                    label = result["label"]
                    score = result["score"]
                    
                    # Add to history
                    if len(st.session_state.analysis_history) >= 10:
                        st.session_state.analysis_history.pop(0)
                    st.session_state.analysis_history.append({"text": user_input[:100] + "..." if len(user_input) > 100 else user_input, 
                                                             "sentiment": label, 
                                                             "score": score})
                    
                    # Display result
                    if label == "POSITIVE":
                        st.success(f"😊 Sentiment: {label}")
                    elif label == "NEGATIVE":
                        st.error(f"😔 Sentiment: {label}")
                    else:
                        st.info(f"😐 Sentiment: {label}")
                    st.write(f"**Confidence**: {score:.3f}")
                    
                    # Visualization
                    fig = px.bar(
                        x=["POSITIVE", "NEUTRAL", "NEGATIVE"],
                        y=[
                            score if label == "POSITIVE" else 0.1,
                            score if label == "NEUTRAL" else 0.1,
                            score if label == "NEGATIVE" else 0.1
                        ],
                        color=["POSITIVE", "NEUTRAL", "NEGATIVE"],
                        labels={"x": "Sentiment", "y": "Confidence"},
                        title="Sentiment Analysis Results"
                    )
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        else:
            st.warning("Please enter some text.")
    
    # Show history
    if st.session_state.analysis_history:
        st.subheader("Recent Analysis History")
        for i, entry in enumerate(reversed(st.session_state.analysis_history)):
            with st.expander(f"#{i+1}: {entry['text']}"):
                sentiment_color = "green" if entry['sentiment'] == "POSITIVE" else "red" if entry['sentiment'] == "NEGATIVE" else "gray"
                st.markdown(f"**Sentiment**: <span style='color:{sentiment_color}'>{entry['sentiment']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence**: {entry['score']:.3f}")

# Batch Analysis Tab
with tabs[1]:
    st.markdown("### Analyze multiple texts at once")
    st.info("Upload a CSV file with a 'text' column containing the texts to analyze.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'text' not in df.columns:
                st.error("CSV file must contain a 'text' column.")
            else:
                if st.button("Run Batch Analysis", key="analyze_batch"):
                    with st.spinner("Analyzing batch data..."):
                        # Process in batches to avoid memory issues
                        batch_size = 100
                        results = []
                        
                        progress_bar = st.progress(0)
                        for i in range(0, len(df), batch_size):
                            batch = df['text'].iloc[i:i+batch_size].tolist()
                            batch_results = [analyze_sentiment(text) for text in batch]
                            results.extend(batch_results)
                            progress_bar.progress(min(1.0, (i + batch_size) / len(df)))
                        
                        # Add results to dataframe
                        df['sentiment'] = [r['label'] for r in results]
                        df['confidence'] = [r['score'] for r in results]
                        
                        # Display results
                        st.subheader("Analysis Results")
                        st.dataframe(df)
                        
                        # Visualization
                        sentiment_counts = df['sentiment'].value_counts().reset_index()
                        sentiment_counts.columns = ['Sentiment', 'Count']
                        
                        fig1 = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                                     title='Sentiment Distribution', color='Sentiment',
                                     color_discrete_map={'POSITIVE': 'green', 'NEUTRAL': 'gray', 'NEGATIVE': 'red'})
                        st.plotly_chart(fig1)
                        
                        # Allow download of results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv",
                        )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Movie Chat Tab
with tabs[2]:
    st.markdown("### Chat about movies with our AI bot!")
    
    if not dialogpt_available:
        st.error(f"DialoGPT model could not be loaded. Error: {dialogpt_error}")
        st.info("You can still use the sentiment analysis features in the other tabs.")
    else:
        chat_input = st.text_input("Your Message", placeholder="E.g., I love sci-fi movies!")
        
        if st.button("Send", key="send_chat"):
            if chat_input.strip():
                with st.spinner("Generating response..."):
                    try:
                        # Analyze sentiment of user input
                        sentiment = analyze_sentiment(chat_input)
                        sentiment_text = f"(Sentiment: {sentiment['label']}, Confidence: {sentiment['score']:.3f})"
                        
                        # Generate bot response
                        inputs = tokenizer.encode(chat_input + tokenizer.eos_token, return_tensors="pt")
                        reply_ids = model.generate(
                            inputs, 
                            max_length=100,  # Shorter to avoid long responses
                            pad_token_id=tokenizer.eos_token_id,
                            do_sample=True,  # Enable sampling for more diverse responses
                            top_p=0.95,      # Nucleus sampling
                            top_k=50         # Top-k sampling
                        )
                        reply = tokenizer.decode(reply_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
                        
                        # Update chat history
                        st.session_state.chat_history.append({
                            "user": chat_input, 
                            "bot": reply, 
                            "sentiment": sentiment_text
                        })
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        st.code(traceback.format_exc())
            else:
                st.warning("Please enter a message.")
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for i, entry in enumerate(st.session_state.chat_history):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"**You**: {entry['user']}")
                    st.caption(entry['sentiment'])
                with col2:
                    st.markdown(f"**Bot**: {entry['bot']}")
                st.divider()
        
        # Tips for chatting
        with st.expander("Tips for chatting"):
            st.markdown("""
            - Ask about movie genres: "What's your favorite sci-fi movie?"
            - Share your opinions: "I think The Godfather is overrated."
            - Ask for recommendations: "Can you recommend a good comedy?"
            - Discuss actors or directors: "Do you like Christopher Nolan films?"
            """)

# Footer
st.markdown("---")
st.markdown("Powered by NLTK VADER Sentiment Analysis and DialoGPT")