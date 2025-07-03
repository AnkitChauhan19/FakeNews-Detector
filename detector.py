import warnings
import numpy as np
import pandas as pd
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

# Loading the Word2Vec model
model = api.load('word2vec-google-news-300')

# Loading cleaned dataset
data = pd.read_csv('clean_data.csv')

# Splitting data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['label'])
train_data = train_data.dropna(subset=['lemm_text'])
test_data = test_data.dropna(subset=['lemm_text'])

# Function to vectorize the text using the model
def vectorize_text(text, model):
    words = text.split()
    word_vectors = [model[word] for word in words if word in model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Extracting predictor variables for train and test sets
X_train = np.array([vectorize_text(text, model) for text in train_data['lemm_text']])
X_test = np.array([vectorize_text(text, model) for text in test_data['lemm_text']])

# Extracting target variable for train and test sets
y_train = train_data['label']
y_test = test_data['label']

# Random Forest model training and prediction
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# Evaluating model
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"RF Accuracy: {accuracy_rf:.4f}")
print("RF Classification Report\n:", classification_report(y_test, y_pred_rf))