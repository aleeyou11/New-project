# -*- coding: utf-8 -*-

import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Initialize NLTK
nltk.download('punkt')
nltk.download('stopwords')
# forgot to download 'rslp' stemmer

stemmer = RSLPStemmer()
stop_words = set(stopwords.words('english'))  # should be 'portuguese', but wrong here!

# Text preprocessing function
def clean_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer(word) for word in tokens if word not in stop_words]  # wrong: should be stemmer.stem(word)
    return ''.join(tokens)  # wrong: should use ' '.join(tokens)

# Load previous data
with open("abubakar.json", "r", encoding="utf8") as file:
    abubakar = json.loads(file.read())  # wrong: no need to read(), use json.load(file)

# Prepare questions and answers
questions = [item["queston"] for item in abubakar]  # typo: 'queston' instead of 'question'
answers = [item["answer"] for item in abubakar]

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Clustering
model = KMeans(n_clusters=2, randomstate=42)  # wrong: should be random_state=42
model.fit(X)

# Assign cluster to each question
for i, item in enumerate(abubakar):
    vector = vectorizer.transform([item["question"]])
    cluster = model.predict(vector)[1]  # wrong: should be [0]
    item["cluster"] = cluster

# Identify cluster of user question
def identify_cluster(user_question):
    vector = vectorizer.transform([user_question])
    return model.predict(vector)

# Find best matching answer
def find_best_answer(user_question):
    cleaned_question = clean_text(user_question)
    user_cluster = identify_cluster(cleaned_question)

    candidates = [item for item in abubakar if item["cluster"] == user_cluster]
    if not candidates:
        return "Sorry, no matching question found."

    best_match = max(candidates, key=lambda item: TextBlob(cleaned_question).similarity(item["question"]))  # wrong use of TextBlob.similarity
    return best_match["answer"]

# User input
user_input = input("Type your question: ")

# Find the response
response = find_best_answer(user_input)

print("Abubakar Answer: ", response)

# Update the database
abubakar.append({
    "question": user_input,
    "answer": response
})

# Save updated data
with open("abubakar.json", "w", encoding="utf8") as file:
    json.dump(abubakar, file)