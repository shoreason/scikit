from sklearn.cluster import KMeans
from sklearn import preprocessing, cross_validation
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd
from matplotlib import pyplot


df = pd.read_csv('data/f3_rb_tot_stats.csv')
print("initial shape ", df.shape)
df.drop(['Player','FPTS'], 1, inplace = True)
# df = df[df.ATT > 100]

print("new shape ", df.shape)


# replace


df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

def condition(value):
    if value >= 20:
        return 'RB1'
    elif value >= 16 and value < 20:
        return 'RB2'
    elif value >= 13 and value < 16:
        return 'RB3'
    elif value >= 10 and value < 13:
        return 'RB4'
    else:
        return 'RB5'

# Assign RB tiers
df['TIER'] = df['FPTS/G'].apply(condition)

def handle_non_numerical_data(df):
    columns = df.columns.values
    # for each column
    for column in columns:
        text_digits_vals = {}
        def convert_to_int(val):
            return text_digits_vals[val]

        # checking if that column is numeric
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            # if not numbers convert column to a list
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            # grab unique elements of list and populate a dictionary
            for unique in unique_elements:
                if unique not in text_digits_vals:
                    text_digits_vals[unique] = x
                    x+=1
            # map values to the column
            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)

print(df.head())

k = 4

X = np.array(df.drop(['TIER'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['TIER'])
clf = KMeans(n_clusters=k)
clf.fit(X)
#
# centroids = clf.cluster_centers_
# labels = clf.labels_

# colors = ["g.","r.", "c.", "b", "y."]
#
#
# for i in range(k):
#     # select only data observations with cluster label == i
#     ds = X[np.where(labels==i)]
#     # plot the data observations
#     pyplot.plot(ds[:,0],ds[:,1],'o')
#     # plot the centroids
#     lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
#     # make the centroid x's bigger
#     pyplot.setp(lines,ms=15.0)
#     pyplot.setp(lines,mew=2.0)
# pyplot.show()


correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    print("prediction ", prediction[0])
    print("actual ", y[i])
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))
