import sys
import os
import time
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing, cross_validation, neighbors, svm
from sklearn.metrics import classification_report, accuracy_score


train = pd.read_csv("data/train.tsv", header=0, delimiter="\t", quoting=3)

num_phrases = train["PhraseId"].size
clean_train_phrases = []
# 0 is negative, 4 is positive
training_sentiment = []

def phrase_to_wordlist(raw_phrase, remove_stopwords=False):
    # make words lowercase
    words = raw_phrase.lower().split()

    # setup stopwords and remove stopwords
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    # pattern = re.compile("^[\w]+$")
    # words = [w for w in words if pattern.match(w) ]
    # words = [w for w in words if len(w)>1 ]

    return(" ".join(words))

sentences = []
last_sentence_id = 0

# split words in sentences and represent each sentence sentence as one hot vector
for i in range(0, num_phrases):
    sentence_id = train["SentenceId"][i]

    if sentence_id != last_sentence_id:
        sentence = phrase_to_wordlist(train["Phrase"][i], remove_stopwords=True)
        # print("sentence: " + str(sentence))
        sentences.append(sentence)
        last_sentence_id = sentence_id
        # print("sentiment: " + str(train["Sentiment"][i]))
        training_sentiment.append(train["Sentiment"][i])





# Create feature vectors
vectorizer = TfidfVectorizer(min_df=5,
                             max_df = 0.8,
                             sublinear_tf=True,
                             use_idf=True)

train_data_features = vectorizer.fit_transform(sentences)
train_data_features = train_data_features.toarray()

# print(sentences)
# print(train_data_features.shape)
#
# vocab = vectorizer.get_feature_names()
# print("vocab", vocab)



X = train_data_features
y = training_sentiment

X_train, X_test, y_train,  y_test = cross_validation.train_test_split(X, y, test_size=0.3)

# clf = neighbors.KNeighborsClassifier()
clf = svm.SVC(kernel="linear")
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print("This model's accuracy is : ", accuracy)

# predicting
# prediction = clf.predict(example_measure)
# print(prediction)
#
# pred = clf.predict(X_test)
# acc = accuracy_score(pred, y_test)
#
# print("Here is how accurate my model is: ",  acc)
