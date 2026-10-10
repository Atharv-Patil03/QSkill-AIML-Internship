import io
import urllib.request
import zipfile
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 1. Download & Extract Dataset directly in code
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
print("Downloading dataset...")
req = urllib.request.urlopen(url)
zip_file = zipfile.ZipFile(io.BytesIO(req.read()))

# Extract and read the TSV file from UCI repository
with zip_file.open("SMSSpamCollection") as f:
    df = pd.read_csv(f, sep="\t", header=None, names=["label", "message"])

# 2. Text Preprocessing & TF-IDF
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

df["clean_message"] = df["message"].apply(
    lambda x: " ".join(
        [word.lower() for word in str(x).split() if word.lower() not in stop_words]
    )
)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_message"])
y = df["label"].map({"ham": 0, "spam": 1})

# 3. Train / Test Split & Model Training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

# 4. Evaluation
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
