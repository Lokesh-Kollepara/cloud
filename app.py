from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import os
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

# File Configuration
MODEL_KEYS = {
    'svm': 'svm_model.pkl',
    'voting': 'voting_model.pkl'
}
VECTORIZER_PATH = 'vectorizer.pkl'

# Check for model and vectorizer files
for model_path in MODEL_KEYS.values():
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found in the current directory.")

if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(f"Vectorizer file {VECTORIZER_PATH} not found in the current directory.")

# Load the models and vectorizer
models = {name: joblib.load(path) for name, path in MODEL_KEYS.items()}
tfidf_vectorizer = joblib.load(VECTORIZER_PATH)

# Define the preprocessing function
def text_preprocessing(text):
    stop_words = set(stopwords.words('english'))
    removed_special_characters = re.sub("[^a-zA-Z]", " ", str(text))
    tokens = removed_special_characters.lower().split()
    stemmer = PorterStemmer()
    cleaned = []
    for token in tokens:
        if token not in stop_words:
            cleaned.append(token)
    stemmed = [stemmer.stem(token) for token in cleaned]
    return " ".join(stemmed)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the review text and model name from the request
        data = request.get_json()
        review_text = data.get('review', '')
        selected_model = data.get('model', 'svm')  # Default to SVM

        # Check if the review text and model are provided
        if not review_text:
            return jsonify({'error': 'No review text provided'}), 400
        if selected_model not in models:
            return jsonify({'error': 'Invalid model selected'}), 400

        # Preprocess the review text
        processed_review = text_preprocessing(review_text)

        # Transform the text using the vectorizer
        transformed_text = tfidf_vectorizer.transform([processed_review])

        # Predict the label using the selected model
        prediction = models[selected_model].predict(transformed_text)
        label = "CG" if prediction[0] == 0 else "OR"

        # Return the prediction as a JSON response
        return jsonify({'prediction': label}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
