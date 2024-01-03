import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import nltk
import random

# Download the words dataset from nltk
nltk.download('words')

#Get the list of English words
english_words = set(nltk.corpus.words.words())

# Function to generate random English words
def generate_random_words(num_words):
    return [random.choice(list(english_words)) for _ in range(num_words)]


# Medical keywords
medical_keywords = ["cancer", "diabetes", "vaccine", "heart disease", "medication", "hospital"]
non_medical_keywords = generate_random_words(len(medical_keywords))


# Create a DataFrame
data = pd.DataFrame({
    'Keyword': medical_keywords + non_medical_keywords,
    'Label': ['Medical'] * len(medical_keywords) + ['Non-Medical'] * len(non_medical_keywords)
})

# Shuffle the DataFrame
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Load Bio Clinical BERT model
model = SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')

# Apply the model to the 'Keyword' column
data['Embeddings'] = data['Keyword'].apply(lambda text: model.encode(text))


print(data)

# Feature extraction
X = list(data['Embeddings'])
y = data['Label']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)


# Evaluate the classifier
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Example predictions
new_keywords = ["flu", "exercise", "headache", "apple", "surgery","non-sense","covid","srini","karuna","renuka","method","generic","tesla"]
# Apply the Bio Clinical BERT model to the new keywords
new_embeddings = [model.encode(keyword) for keyword in new_keywords]
predictions = classifier.predict(new_embeddings)

result = pd.DataFrame({'Keyword': new_keywords, 'Prediction': predictions})
print(result)