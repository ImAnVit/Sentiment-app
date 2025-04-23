import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Print NLTK version and data path
print(f'NLTK version: {nltk.__version__}')
print(f'NLTK data path: {nltk.data.path}')

# Try to use the VADER sentiment analyzer
try:
    analyzer = SentimentIntensityAnalyzer()
    test_text = "This is a test. NLTK is working properly!"
    result = analyzer.polarity_scores(test_text)
    print(f'VADER test result: {result}')
    print('NLTK and VADER are working properly!')
except Exception as e:
    print(f'Error: {str(e)}')
