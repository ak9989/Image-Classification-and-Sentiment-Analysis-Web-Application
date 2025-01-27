import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle  # or joblib if you decide to use it

# Load the dataset
data = pd.read_csv("IMDB_reviews.csv")

# Ensure all values in 'Rating By User' are strings
data['Rating By User'] = data['Rating By User'].astype(str)

# Remove rows with invalid 'Rating By User' values (e.g., 'nan', empty strings, etc.)
data = data[data['Rating By User'].str.contains(r'^\d+/10$', na=False)]

# Convert 'Rating By User' to binary sentiment labels (1 for ratings >= 7, 0 for others)
y = data['Rating By User'].apply(lambda x: 1 if int(x.split('/')[0]) >= 7 else 0)

# Extract the 'Review Text' column as input (X)
data = data.dropna(subset=['Review Text'])  # Remove rows with missing reviews
X = data['Review Text']

# Preprocess text using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
model.fit(X_train, y_train)

# Save the model and vectorizer using pickle (or joblib)
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))

# If you want to use joblib instead, you can use the following:
# import joblib
# joblib.dump(model, "sentiment_model.joblib")
# joblib.dump(tfidf, "tfidf_vectorizer.joblib")

print("Model and vectorizer saved successfully!")
