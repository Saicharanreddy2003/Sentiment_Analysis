# sentiment_analysis_imdb.py

import os
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def preprocess_text(text):
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase the text
    text = text.lower()
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word.isalnum() and word not in stopwords.words('english')]
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

    df = pd.DataFrame(data)
    return df

# Provide the path to the extracted IMDb dataset on your desktop
# Change this line based on your VS Code project folder structure
# Assuming "sentiment_analysis" is your VS Code project folder
vscode_project_folder = 'sentiment_analysis'
imdb_path = os.path.join(os.path.expanduser('~'), 'Desktop', vscode_project_folder, 'aclImdb_v1')




# Load IMDb dataset into a Pandas DataFrame
imdb_df = load_imdb_dataset(imdb_path)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(imdb_df['text'])
y = imdb_df['sentiment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Function for user input and prediction
def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = classifier.predict(vectorized_text)[0]
    return prediction

# Example usage
user_input = input("Enter a text for sentiment analysis: ")
result = predict_sentiment(user_input)
print(f"Predicted Sentiment: {result}")
