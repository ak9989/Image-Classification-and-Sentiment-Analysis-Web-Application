from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
sentiment_model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sentiment-analysis', methods=['POST'])
def analyze_sentiment():
    text = request.form['text']
    vectorized_text = tfidf_vectorizer.transform([text])
    prediction = sentiment_model.predict(vectorized_text)
    sentiment = "positive" if prediction[0] == 1 else "negative"
    return render_template("index.html", prediction_text=f"The sentiment is {sentiment}")

if __name__ == "__main__":
    app.run(debug=True)


