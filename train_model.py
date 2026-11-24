"""Train a small educational movie-review sentiment classifier."""
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Small, manually written teaching dataset. Not a production benchmark.
positive_reviews = [
    "I loved this movie, it was fantastic and exciting",
    "A wonderful film with brilliant acting",
    "The story was beautiful and very moving",
    "Absolutely amazing, I would watch it again",
    "Great performances and an excellent script",
    "The film was funny, entertaining and charming",
    "What an incredible movie, I highly recommend it",
    "I enjoyed every moment of this brilliant film",
    "A delightful story with a satisfying ending",
    "The actors were superb and the direction was excellent",
    "One of the best films I have seen this year",
    "This movie was inspiring and full of heart",
    "The plot was clever and the characters were lovable",
    "A truly memorable and uplifting experience",
    "I was impressed by the beautiful cinematography",
    "The soundtrack was gorgeous and the acting was strong",
    "Such a fun and enjoyable movie",
    "An outstanding film that exceeded my expectations",
    "I was captivated from beginning to end",
    "This film was not boring at all",
    "The performances were powerful and convincing",
    "A fantastic adventure with exciting moments",
    "The movie made me smile and laugh",
    "The ending was perfect and emotional",
    "A lovely movie with a great message",
    "The story kept me interested throughout",
    "Wonderful characters and excellent dialogue",
    "I really liked the pacing and the humour",
    "This film is worth watching, very enjoyable",
    "The film was surprisingly good and entertaining",
    "Not bad at all, actually a very good movie",
    "The movie was creative, fresh and engaging",
    "I loved the visuals and the original story",
    "Excellent writing and unforgettable performances",
    "The film was touching and beautifully directed",
    "A superb movie with a fantastic cast",
    "I was thrilled by the suspense and excitement",
    "A feel good film with wonderful music",
    "The ending was satisfying and well written",
    "A highly entertaining film I would recommend",
]

negative_reviews = [
    "I hated this movie, it was terrible and boring",
    "A disappointing film with awful acting",
    "The story was confusing and very weak",
    "Absolutely dreadful, I would never watch it again",
    "Poor performances and a terrible script",
    "The film was dull, annoying and predictable",
    "What a horrible movie, I do not recommend it",
    "I regretted watching this badly made film",
    "A messy story with an unsatisfying ending",
    "The actors were unconvincing and the direction was poor",
    "One of the worst films I have seen this year",
    "This movie was irritating and completely pointless",
    "The plot was silly and the characters were unlikeable",
    "A truly forgettable and frustrating experience",
    "I was disappointed by the ugly cinematography",
    "The soundtrack was irritating and the acting was weak",
    "Such a dull and unpleasant movie",
    "A terrible film that failed my expectations",
    "I was bored from beginning to end",
    "This film was not enjoyable at all",
    "The performances were wooden and unconvincing",
    "A tedious adventure with no exciting moments",
    "The movie made me angry and annoyed",
    "The ending was awful and disappointing",
    "A bad movie with a ridiculous message",
    "The story never kept me interested",
    "Weak characters and terrible dialogue",
    "I really disliked the pacing and the humour",
    "This film is not worth watching, very disappointing",
    "The film was surprisingly bad and uninteresting",
    "Not good at all, actually a very poor movie",
    "The movie was repetitive, stale and uninspired",
    "I disliked the visuals and the predictable story",
    "Terrible writing and forgettable performances",
    "The film was annoying and poorly directed",
    "An awful movie with a weak cast",
    "I was frustrated by the slow and boring plot",
    "A depressing film with unpleasant music",
    "The ending was confusing and badly written",
    "An extremely boring film I would avoid",
]

# Add varied, labelled synthetic examples to teach common sentiment vocabulary.
# These are generated from templates, NOT real audience reviews.
positive_words = [
    "amazing", "brilliant", "excellent", "wonderful", "fantastic",
    "enjoyable", "beautiful", "engaging", "outstanding", "delightful",
    "superb", "heartwarming", "impressive", "funny", "exciting",
    "charming", "memorable", "inspiring", "entertaining", "great"
]
negative_words = [
    "awful", "terrible", "boring", "dreadful", "disappointing",
    "annoying", "confusing", "weak", "unpleasant", "tedious",
    "predictable", "forgettable", "frustrating", "messy", "horrible",
    "poor", "ridiculous", "uninspired", "irritating", "bad"
]
subjects = ["movie", "film", "story", "acting", "plot", "ending",
            "dialogue", "direction", "soundtrack", "cinematography"]
templates = [
    "The {subject} was {word}.",
    "I thought the {subject} was really {word}.",
    "In my opinion, the {subject} is {word}.",
    "Overall, the {subject} felt {word}.",
]
for index, word in enumerate(positive_words):
    for subject in subjects:
        positive_reviews.append(
            templates[index % len(templates)].format(subject=subject, word=word)
        )
for index, word in enumerate(negative_words):
    for subject in subjects:
        negative_reviews.append(
            templates[index % len(templates)].format(subject=subject, word=word)
        )

reviews = positive_reviews + negative_reviews
labels = ["positive"] * len(positive_reviews) + ["negative"] * len(negative_reviews)

X_train, X_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.25, random_state=42, stratify=labels
)

# Pipeline saves both TF-IDF text preprocessing and the classifier.
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, predictions):.1%}")
print(classification_report(y_test, predictions, zero_division=0))

model_path = Path(__file__).resolve().parent / "sentiment_model.joblib"
joblib.dump(model, model_path)
print(f"Saved model to: {model_path}")
