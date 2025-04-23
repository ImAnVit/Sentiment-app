import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random
import re

# Set page title and configuration
st.set_page_config(page_title="Movie Chat & Sentiment App", layout="wide")

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

# Simple rule-based movie chatbot
class MovieChatbot:
    def __init__(self):
        self.movie_genres = ["action", "comedy", "drama", "horror", "sci-fi", "thriller", 
                            "romance", "animation", "documentary", "fantasy", "adventure"]
        
        self.popular_movies = {
            "action": ["Die Hard", "The Matrix", "John Wick", "Mad Max: Fury Road", "The Dark Knight"],
            "comedy": ["Superbad", "Anchorman", "Bridesmaids", "The Hangover", "Step Brothers"],
            "drama": ["The Shawshank Redemption", "The Godfather", "Schindler's List", "Forrest Gump", "The Green Mile"],
            "horror": ["The Shining", "Hereditary", "Get Out", "The Exorcist", "A Quiet Place"],
            "sci-fi": ["Blade Runner", "Interstellar", "The Martian", "Arrival", "Dune"],
            "thriller": ["Silence of the Lambs", "Se7en", "Parasite", "Gone Girl", "Shutter Island"],
            "romance": ["The Notebook", "Pride and Prejudice", "La La Land", "Before Sunrise", "Eternal Sunshine of the Spotless Mind"],
            "animation": ["Toy Story", "Spirited Away", "Spider-Man: Into the Spider-Verse", "The Lion King", "WALL-E"],
            "documentary": ["March of the Penguins", "Free Solo", "13th", "Won't You Be My Neighbor?", "My Octopus Teacher"],
            "fantasy": ["The Lord of the Rings", "Harry Potter", "Pan's Labyrinth", "The Princess Bride", "The Shape of Water"],
            "adventure": ["Indiana Jones", "The Goonies", "Pirates of the Caribbean", "Jurassic Park", "The Mummy"]
        }
        
        self.famous_directors = ["Christopher Nolan", "Steven Spielberg", "Martin Scorsese", 
                                "Quentin Tarantino", "James Cameron", "Greta Gerwig", 
                                "Alfred Hitchcock", "Stanley Kubrick", "Francis Ford Coppola"]
        
        self.famous_actors = ["Tom Hanks", "Meryl Streep", "Leonardo DiCaprio", "Denzel Washington", 
                            "Viola Davis", "Robert De Niro", "Jennifer Lawrence", "Brad Pitt", 
                            "Cate Blanchett", "Morgan Freeman"]
        
        # Patterns for matching user inputs
        self.patterns = {
            "greeting": r"(?i)(\bhello\b|\bhi\b|\bhey\b|\bgreetings\b)",
            "how_are_you": r"(?i)(how are you|how's it going|how do you do|what's up)",
            "recommend": r"(?i)(recommend|suggest|what should I watch|good movie)",
            "genre": r"(?i)(action|comedy|drama|horror|sci-fi|thriller|romance|animation|documentary|fantasy|adventure)",
            "opinion": r"(?i)(what (?:do you think|is your opinion) about|have you seen|do you like)",
            "director": r"(?i)(director|filmmaker|made by)",
            "actor": r"(?i)(actor|actress|star|starring|plays in)",
            "best": r"(?i)(best|greatest|favorite|top|excellent)",
            "worst": r"(?i)(worst|terrible|bad|awful|dislike)",
            "thank": r"(?i)(thank|thanks|appreciate)",
            "bye": r"(?i)(bye|goodbye|see you|farewell)"
        }
        
        # Responses for different patterns
        self.responses = {
            "greeting": [
                "Hello! I'm your friendly movie chatbot. What kind of movies do you enjoy?",
                "Hi there! Ready to talk about some great films?",
                "Hey! I'm excited to chat about movies with you today!"
            ],
            "how_are_you": [
                "I'm doing great, thanks for asking! Always happy to discuss movies. What's on your mind?",
                "I'm excellent! I've been thinking about movies all day. What can I help you with?",
                "All good here in the digital world! Ready to talk about your favorite films?"
            ],
            "recommend_generic": [
                "I'd recommend checking out 'The Shawshank Redemption' - it's a classic for a reason!",
                "Have you seen 'Parasite'? It's a brilliant film that won Best Picture!",
                "I think 'Everything Everywhere All at Once' is a must-watch if you haven't seen it yet.",
                "You might enjoy 'Inception' - it's a mind-bending experience!",
                "How about 'The Grand Budapest Hotel'? It's visually stunning and has a great story."
            ],
            "opinion_generic": [
                "That's a fascinating film with some really memorable moments!",
                "I think it's a solid movie that showcases some great performances.",
                "It's definitely one that makes you think! The cinematography is excellent too.",
                "That one has some devoted fans! The direction is particularly noteworthy."
            ],
            "director_generic": [
                "They've made some incredible contributions to cinema! Which of their films have you seen?",
                "One of the most influential directors of their generation! Do you have a favorite film by them?",
                "Their visual style is so distinctive! I always look forward to their new releases."
            ],
            "actor_generic": [
                "They're incredibly talented! They bring so much depth to their characters.",
                "One of the finest performers working today! Have you seen their recent work?",
                "They have such an impressive range! They can do comedy and drama equally well."
            ],
            "best_generic": [
                "It's hard to pick the absolute best, but 'The Godfather' is often considered one of the greatest films ever made.",
                "Many critics would say 'Citizen Kane' changed cinema forever and remains one of the best.",
                "For pure entertainment value, it's hard to beat the original 'Star Wars'!"
            ],
            "worst_generic": [
                "There are quite a few contenders for that title! 'The Room' is famously so bad it's good.",
                "Some of the movies with 0% on Rotten Tomatoes might qualify!",
                "Everyone has different tastes - one person's worst movie might be another's guilty pleasure!"
            ],
            "thank": [
                "You're welcome! I'm happy to talk movies anytime.",
                "My pleasure! Is there anything else you'd like to discuss about cinema?",
                "Glad I could help! Let me know if you want to chat more about films."
            ],
            "bye": [
                "Goodbye! Enjoy your movie watching!",
                "See you later! Hope you find something great to watch!",
                "Until next time! The credits may be rolling on our conversation, but there's always a sequel!"
            ],
            "default": [
                "Interesting perspective on movies! What genres do you typically enjoy?",
                "I'm not quite sure I understood that. Could you tell me more about your movie preferences?",
                "That's a unique take! Have you seen any good films lately?",
                "Movies are such a rich topic for discussion! Is there a particular era of film you enjoy most?"
            ]
        }
    
    def get_response(self, user_input):
        # Check for patterns in user input
        if re.search(self.patterns["greeting"], user_input):
            return random.choice(self.responses["greeting"])
        
        elif re.search(self.patterns["how_are_you"], user_input):
            return random.choice(self.responses["how_are_you"])
        
        elif re.search(self.patterns["thank"], user_input):
            return random.choice(self.responses["thank"])
        
        elif re.search(self.patterns["bye"], user_input):
            return random.choice(self.responses["bye"])
        
        elif re.search(self.patterns["recommend"], user_input):
            # Check if a specific genre is mentioned
            genre_match = re.search(self.patterns["genre"], user_input)
            if genre_match:
                genre = genre_match.group(0).lower()
                if genre in self.movie_genres:
                    movies = self.popular_movies.get(genre, [])
                    if movies:
                        movie = random.choice(movies)
                        return f"For {genre}, I'd recommend '{movie}'. It's one of my favorites in that genre!"
            
            # Generic recommendation
            return random.choice(self.responses["recommend_generic"])
        
        elif re.search(self.patterns["opinion"], user_input):
            # Generic opinion
            return random.choice(self.responses["opinion_generic"])
        
        elif re.search(self.patterns["director"], user_input):
            # Check if a specific director is mentioned
            for director in self.famous_directors:
                if director.lower() in user_input.lower():
                    return f"{director} is a brilliant filmmaker! Their visual storytelling and attention to detail are remarkable."
            
            # Generic director response
            return random.choice(self.responses["director_generic"])
        
        elif re.search(self.patterns["actor"], user_input):
            # Check if a specific actor is mentioned
            for actor in self.famous_actors:
                if actor.lower() in user_input.lower():
                    return f"{actor} brings such authenticity to every role. Their performances are always captivating!"
            
            # Generic actor response
            return random.choice(self.responses["actor_generic"])
        
        elif re.search(self.patterns["best"], user_input):
            # Check if a specific genre is mentioned for "best"
            genre_match = re.search(self.patterns["genre"], user_input)
            if genre_match:
                genre = genre_match.group(0).lower()
                if genre in self.movie_genres and genre in self.popular_movies:
                    movie = self.popular_movies[genre][0]  # First movie in the list (presumably the best)
                    return f"For {genre}, '{movie}' is widely considered one of the best!"
            
            # Generic "best" response
            return random.choice(self.responses["best_generic"])
        
        elif re.search(self.patterns["worst"], user_input):
            return random.choice(self.responses["worst_generic"])
        
        # Default response if no patterns match
        return random.choice(self.responses["default"])

# Initialize chatbot
chatbot = MovieChatbot()

# Initialize session states
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🎬 Movie Chat & Sentiment Analysis")
st.markdown("### Talk about movies with our AI assistant and analyze sentiment")

# Create tabs with Movie Chat as the first and main option
tabs = st.tabs(["Movie Chat", "Text Analysis", "Batch Analysis"])

# Movie Chat Tab (now first)
with tabs[0]:
    st.header("💬 Movie Chat")
    
    # Create a more prominent chat interface
    st.markdown("""
    <style>
    .chat-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Chat input area
    chat_input = st.text_input("Your Message", placeholder="E.g., I love sci-fi movies! What do you recommend?")
    col1, col2 = st.columns([1, 5])
    with col1:
        send_button = st.button("Send", key="send_chat", use_container_width=True)
    with col2:
        if st.button("Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Process chat input
    if send_button and chat_input.strip():
        with st.spinner("Generating response..."):
            try:
                # Generate bot response using rule-based chatbot
                reply = chatbot.get_response(chat_input)
                
                # Update chat history without sentiment analysis
                st.session_state.chat_history.append({
                    "user": chat_input, 
                    "bot": reply
                })
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    
    # Display chat history with larger text and better structure
    if st.session_state.chat_history:
        # Add custom CSS for chat bubbles
        st.markdown("""
        <style>
        .chat-container {
            margin-bottom: 30px;
            max-width: 100%;
        }
        .user-message {
            background-color: #2b5797;
            color: white;
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 10px;
            font-size: 18px;
            font-weight: 500;
        }
        .bot-message {
            background-color: #0078d4;
            color: white;
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 20px;
            font-size: 18px;
            font-weight: 500;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create a container for the chat
        for i, entry in enumerate(st.session_state.chat_history):
            # User message
            st.markdown(f'<div class="user-message"><strong>You:</strong> {entry["user"]}</div>', unsafe_allow_html=True)
            
            # Bot message
            st.markdown(f'<div class="bot-message"><strong>Movie Bot:</strong> {entry["bot"]}</div>', unsafe_allow_html=True)
    else:
        st.info("Start chatting with the Movie Bot! Ask about movie recommendations, share your opinions, or discuss your favorite films.")
    
    # Tips for chatting
    with st.expander("Tips for better conversations"):
        st.markdown("""
        - **Ask about genres**: "What's your favorite sci-fi movie?"
        - **Share opinions**: "I think The Godfather is overrated."
        - **Get recommendations**: "Can you recommend a good comedy?"
        - **Discuss directors**: "Do you like Christopher Nolan films?"
        - **Talk about classics**: "What makes Citizen Kane so important?"
        - **Mention actors**: "I love movies with Tom Hanks."
        """)

# Text Analysis Tab (now second)
with tabs[1]:
    st.header("📊 Text Sentiment Analysis")
    st.markdown("Analyze the sentiment of your text.")
    
    user_input = st.text_area("Input Text", placeholder="Type your text here...", height=150)
    
    # Make buttons the same size
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        analyze_button = st.button("Analyze Sentiment", key="analyze_single", use_container_width=True)
    with col2:
        if st.button("Clear Analysis History", key="clear_analysis", use_container_width=True):
            st.session_state.analysis_history = []
            st.rerun()
    with col3:
        pass  # Empty column to push buttons to the left
    
    if analyze_button:
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

# Batch Analysis Tab (now third)
with tabs[2]:
    st.header("📈 Batch Sentiment Analysis")
    st.markdown("Analyze multiple texts at once.")
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

# Footer
st.markdown("---")
st.markdown("Powered by NLTK VADER Sentiment Analysis")