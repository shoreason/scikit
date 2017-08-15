from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import preprocessing, cross_validation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd
from matplotlib import pyplot
from itertools import cycle


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


df = pd.read_csv('data/f3_rb_tot_stats.csv')
# Assign RB tiers
df['TIER'] = df['FPTS/G'].apply(condition)
original_df = pd.DataFrame.copy(df)
df.drop(['Player', 'OWN', 'FL', 'G'], 1, inplace = True)
df = df[(df.ATT > 10)]
# df = df[(df.G > 4) & (df.ATT > 100)]
# replace
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)





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



X = np.array(df.drop(['TIER'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['TIER'])

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

clf = MeanShift(bandwidth=bandwidth, bin_seeding=True)
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

# add a new column
original_df['cluster_group'] = np.nan
for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))
rb1_rates = {}



for i in range(n_clusters_):
    temp_df = original_df[(original_df['cluster_group']==float(i))]
    rb_cluster = temp_df[(temp_df['TIER']=='RB1')]
    rb1_rate = len(rb_cluster)/len(temp_df)
    rb1_rates[i] = rb1_rate

print(rb1_rates)


print("number of estimated clusters : %d" % n_clusters_)

print(original_df[(original_df['cluster_group']==0)])

print(original_df[(original_df['cluster_group']==1)])

print(original_df[(original_df['cluster_group']==2)])
#
# print("group 0")
# print(original_df[(original_df['cluster_group']==0)].describe())
# print("group 1")
# print(" ")
# print(original_df[(original_df['cluster_group']==1)].describe())
# print("group 2")
# print(" ")
# print(original_df[(original_df['cluster_group']==2)].describe())

centroids = clf.cluster_centers_
labels = clf.labels_

# colors = 10*['r','g','b','c','k','y','m']
#
# print(colors)
# print(labels)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# for i in range(len(X)):
#
#     ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')
#
#
# ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],
#             marker="x",color='k', s=150, linewidths = 5, zorder=10)
#
# plt.show()
