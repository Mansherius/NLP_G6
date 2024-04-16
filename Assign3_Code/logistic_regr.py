import pandas as pd
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Step 1: Data Loading and Preprocessing
data = pd.read_csv(r'.\mbti_1.csv', encoding='latin1', error_bad_lines=False)

# Splitting the posts
data['posts'] = data['posts'].apply(lambda x: x.split('|||'))

# Step 2: Feature Engineering
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(data['posts'].apply(lambda x: ' '.join(x)))

# Step 3: Model Training and Evaluation
classifier = LogisticRegression(max_iter=1000)
y = data['type']

# 5-fold cross-validation
predicted = cross_val_predict(classifier, X, y, cv=5)

# Step 4: Performance Evaluation
accuracy = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
precision = cross_val_score(classifier, X, y, cv=5, scoring='precision_macro')
recall = cross_val_score(classifier, X, y, cv=5, scoring='recall_macro')
f1 = cross_val_score(classifier, X, y, cv=5, scoring='f1_macro')

# Print results
print("Accuracy:", accuracy.mean())
print("Precision:", precision.mean())
print("Recall:", recall.mean())
print("F1-score:", f1.mean())

# Confusion Matrix
conf_matrix = confusion_matrix(y, predicted)
print("Confusion Matrix:")
print(conf_matrix)