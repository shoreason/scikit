import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors, svm
from sklearn.metrics import accuracy_score

# Number of Instances: 690
# Number of Attributes: 15 + class attribute
# +: 307 (44.5%)
# -: 383 (55.5%)
df = pd.read_csv('data/credit-app.data.txt')


dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9,
    'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18,
    's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, 'aa': 27,
    'bb': 28, 'cc': 29, 'dd': 30, 'ee': 31, 'ff': 32, 'gg': 33 }

# replace
df['a1'] = df['a1'].replace('?', 'z')
df['a2'] = df['a2'].replace('?', -99999)
df['a3'] = df['a3'].replace('?', -99999)
df['a4'] = df['a4'].replace('?', 'z')
df['a5'] = df['a5'].replace('?', 'z')
df['a6'] = df['a6'].replace('?', 'z')
df['a7'] = df['a7'].replace('?', 'z')
df['a8'] = df['a8'].replace('?', -99999)
df['a9'] = df['a9'].replace('?', 'z')
df['a10'] = df['a10'].replace('?', 'z')
df['a11'] = df['a11'].replace('?', -99999)
df['a12'] = df['a12'].replace('?', 'z')
df['a13'] = df['a13'].replace('?', 'z')
df['a14'] = df['a14'].replace('?', -99999)
df['a15'] = df['a15'].replace('?', -99999)

df.replace('a', dict['a'], inplace=True)
df.replace('b', dict['b'], inplace=True)
df.replace('c', dict['c'], inplace=True)
df.replace('d', dict['d'], inplace=True)
df.replace('e', dict['e'], inplace=True)
df.replace('f', dict['f'], inplace=True)
df.replace('g', dict['g'], inplace=True)
df.replace('h', dict['h'], inplace=True)
df.replace('i', dict['i'], inplace=True)
df.replace('j', dict['j'], inplace=True)
df.replace('k', dict['k'], inplace=True)
df.replace('l', dict['l'], inplace=True)
df.replace('m', dict['m'], inplace=True)
df.replace('n', dict['n'], inplace=True)
df.replace('o', dict['o'], inplace=True)
df.replace('p', dict['p'], inplace=True)
df.replace('q', dict['q'], inplace=True)
df.replace('r', dict['r'], inplace=True)
df.replace('s', dict['s'], inplace=True)
df.replace('t', dict['t'], inplace=True)
df.replace('u', dict['u'], inplace=True)
df.replace('v', dict['v'], inplace=True)
df.replace('w', dict['w'], inplace=True)
df.replace('x', dict['x'], inplace=True)
df.replace('y', dict['y'], inplace=True)
df.replace('z', dict['z'], inplace=True)
df.replace('aa', dict['aa'], inplace=True)
df.replace('bb', dict['bb'], inplace=True)
df.replace('cc', dict['cc'], inplace=True)
df.replace('dd', dict['dd'], inplace=True)
df.replace('ee', dict['ee'], inplace=True)
df.replace('ff', dict['ff'], inplace=True)
df.replace('gg', dict['gg'], inplace=True)



print(df)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train,  y_test = cross_validation.train_test_split(X, y, test_size=0.3)

clf = neighbors.KNeighborsClassifier()
# clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print("This model's accuracy is : ", accuracy)

example_measure = np.array([6,  12.25,   8.000,  10,   3,   4,  16,   0.335,  10,    6,    0,    6,   19,  150,  245])

example_measure = example_measure.reshape(1, -1)

prediction = clf.predict(example_measure)
print(prediction)

pred = clf.predict(X_test)
acc = accuracy_score(pred, y_test)

print("Here is how accurate my model is: ",  acc)
