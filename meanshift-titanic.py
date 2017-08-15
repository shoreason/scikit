# https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
from sklearn.cluster import MeanShift
from sklearn import preprocessing, cross_validation
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd


'''
Pclass passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1= Yes)
name Name
sex Sex
age Age
sibsp Number of siblings/spouses aboard
parch Number of Parents/children aboard
ticket Ticket Number
fare Passenger Fare (british pound)
cabin Cabin
embarked Port of Embarkation (C = cherbourg; Q = Queenstown ; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

df = pd.read_excel('data/titanic.xls')
original_df = pd.DataFrame.copy(df)
df.drop(['body', 'name'], 1, inplace = True)
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
#
# print(df.head())

df.drop(['boat'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

# add a new column
original_df['cluster_group'] = np.nan

for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))
survival_rates = {}


for i in range(n_clusters_):
    temp_df = original_df[(original_df['cluster_group']==float(i))]
    survival_cluster = temp_df[(temp_df['survived']==1)]
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)

print("group 0")
print(original_df[(original_df['cluster_group']==0)].describe())
print("group 1")
print(" ")
print(original_df[(original_df['cluster_group']==1)].describe())
print("group 2")
print(" ")
print(original_df[(original_df['cluster_group']==2)].describe())
print("group 3")
print(" ")
print(original_df[(original_df['cluster_group']==3)].describe())
