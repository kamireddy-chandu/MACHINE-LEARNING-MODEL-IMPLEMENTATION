import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = {
    'text': [
        'Congratulations! You have won a free ticket!',
        'Call now to claim your prize',
        'Hey, are we meeting today?',
        'Win a free mobile recharge now',
        'Are you coming to the class?',
        'Get cash now!!! Limited offer!',
        'Don’t forget to bring your notebook',
        'Exclusive deal just for you. Click here!',
        'Let’s have dinner at 8?',
        'Claim your free prize money now'
    ],
    'label': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 1 = Spam, 0 = Ham
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


while True:
    msg = input("Enter a message to classify (or 'quit'): ")
    if msg.lower() == 'quit':
        break
    msg_vector = vectorizer.transform([msg])
    prediction = model.predict(msg_vector)[0]
    print("=> Spam" if prediction == 1 else "=> Not Spam")
