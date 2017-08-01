# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd
import re
import nltk
# nltk.download()
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing, cross_validation, neighbors, svm
from sklearn.metrics import classification_report, accuracy_score



train = pd.read_csv("data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)


# view data shape
print(train.shape)


def review_to_words( raw_review ):
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()

    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    words = letters_only.lower().split()

    stops = set(stopwords.words("english"))

    meaningful_words = [w for w in words if not w in stops]

    return( " ".join( meaningful_words ))


num_reviews = train["review"].size
print("number of reviews ", num_reviews)
clean_train_reviews = []

for i in range( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append( review_to_words( train["review"][i] ) )

# print("All cleaned reviews", clean_train_reviews)

# vectorizer = CountVectorizer(analyzer = "word",   \
#                              tokenizer = None,    \
#                              preprocessor = None, \
#                              stop_words = None,   \
#                              max_features = 5000)


vectorizer = TfidfVectorizer(min_df=5,
                             max_df = 0.8,
                             sublinear_tf=True,
                             use_idf=True)

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

# print(train_data_features.shape)
#
# vocab = vectorizer.get_feature_names()
# print("vocab", vocab)

X = train_data_features
y = train["sentiment"]

X_train, X_test, y_train,  y_test = cross_validation.train_test_split(X, y, test_size=0.3)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print("This model's accuracy is : ", accuracy)
