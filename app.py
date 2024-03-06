# app.py

from flask import Flask, render_template, request
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

app = Flask(__name__)

def preprocess_text(text):
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase the text
    text = text.lower()
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word.isalnum()]
    return ' '.join(words)

def load_imdb_dataset(data_path):
    data = {'text': [], 'sentiment': []}

    try:
        # Load positive reviews
        pos_folder = os.path.join(data_path, 'train', 'pos')
        for filename in os.listdir(pos_folder):
            file_path = os.path.join(pos_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                data['text'].append(text)
                data['sentiment'].append('positive')

        # Load negative reviews
        neg_folder = os.path.join(data_path, 'train', 'neg')
        for filename in os.listdir(neg_folder):
            file_path = os.path.join(neg_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                data['text'].append(text)
                data['sentiment'].append('negative')
    except FileNotFoundError:
        print(f"Error: The system cannot find the specified path: {data_path}")
        exit(1)

    return data

vscode_project_folder = 'sentiment_analysis'
imdb_path = os.path.join(os.path.expanduser('~'), 'Desktop', vscode_project_folder, 'aclImdb_v1')


# Load IMDb dataset
imdb_data = load_imdb_dataset(imdb_path)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(imdb_data['text'])
y = imdb_data['sentiment']

# Initialize and train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        user_input = request.form['user_input']
        cleaned_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = classifier.predict(vectorized_text)[0]
        result = f"Predicted Sentiment: {prediction}"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
