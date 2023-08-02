from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

app = Flask(__name__)

# Preprocess the text
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stop_words = ENGLISH_STOP_WORDS

def preprocess_text(text):
    tokens = text.lower().split()
    words = [word for word in tokens if word not in stop_words]
    return ' '.join(words)

# Load the pre-trained model and vectorizer
model = SVC(kernel='linear', C=1.0)
vectorizer = TfidfVectorizer(max_features=5000)

# Load the dataset and train the model (assuming 'df' and 'y' are defined earlier)
data_path = 'datasets/Hate_Speech_and_Offensive_Language_Dataset.csv'
df = pd.read_csv(data_path)
df.drop(columns=['Unnamed: 0'], inplace=True)
df.dropna(inplace=True)
df['processed_tweet'] = df['tweet'].apply(preprocess_text)
X = vectorizer.fit_transform(df['processed_tweet']).toarray()
y = df['class']
model.fit(X, y)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle text classification
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['comment']
    if not data:
        return jsonify({'error': 'Please provide a comment.'})
    
    processed_data = preprocess_text(data)
    X_test = vectorizer.transform([processed_data]).toarray()
    prediction = model.predict(X_test)[0]
    
    # Convert prediction class to the actual label (modify as per your dataset's labels)
    labels = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    predicted_class = labels.get(prediction, 'Unknown')
    
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
