from flask import Flask, request, jsonify, render_template
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample FAQ knowledge base for Kawasaki super bikes
FAQS = {
    "What is the warranty period for Kawasaki super bikes?": "Kawasaki super bikes come with a two-year warranty period.",
    "How do I reset the ECU on my Kawasaki super bike?": "To reset the ECU, turn the ignition on, hold the reset button for 10 seconds, then turn the ignition off.",
    "Can I use my Kawasaki super bike internationally?": "Yes, Kawasaki super bikes are designed to be used internationally. Make sure to check the local regulations.",
    "What should I do if my Kawasaki super bike is not starting?": "If your Kawasaki super bike is not starting, check the battery and fuel levels, then contact our support team if the issue persists.",
    "Is there a mobile app for Kawasaki super bikes?": "Yes, you can download the Kawasaki Rideology app from the App Store or Google Play.",
    "Where can I find the nearest Kawasaki service center?": "You can find the nearest Kawasaki service center by visiting our official website and using the service center locator.",
    "What is the recommended tire pressure for Kawasaki super bikes?": "The recommended tire pressure for Kawasaki super bikes is 36 psi for the front and 42 psi for the rear.",
    "How often should I service my Kawasaki super bike?": "It is recommended to service your Kawasaki super bike every 6,000 miles or every 6 months, whichever comes first.",
    "What type of fuel should I use for my Kawasaki super bike?": "Use premium unleaded gasoline with an octane rating of 91 or higher for optimal performance.",
    "Can I customize my Kawasaki super bike with aftermarket parts?": "Yes, you can customize your Kawasaki super bike with aftermarket parts. However, make sure they are compatible with your specific model."
}

# Preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

# Prepare the FAQ data
faq_keys = list(FAQS.keys())
preprocessed_faq_keys = [preprocess_text(key) for key in faq_keys]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer().fit(preprocessed_faq_keys)
faq_vectors = vectorizer.transform(preprocessed_faq_keys)

# Function to find the best matching FAQ
def find_best_match(question):
    question = preprocess_text(question)
    question_vector = vectorizer.transform([question])
    similarity_scores = cosine_similarity(question_vector, faq_vectors)
    best_match_index = similarity_scores.argmax()
    return faq_keys[best_match_index], FAQS[faq_keys[best_match_index]]

# Function to handle chatbot response
def chatbot_response(question):
    question, answer = find_best_match(question)
    satisfactory = "not satisfied" not in question.lower()
    return answer, satisfactory

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    question = data.get('question', '')
    answer, satisfactory = chatbot_response(question)
    return jsonify({'answer': answer, 'satisfactory': satisfactory})

if __name__ == '__main__':
    app.run(debug=True)
