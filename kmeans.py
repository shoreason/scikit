from sklearn import cluster, datasets
from matplotlib import pyplot
import numpy as np

k = 2

iris = datasets.load_iris()
# X_iris = iris.data
X_iris = np.array([98, 80 , 78, 10, 5, 20, 46, 55, 60])
y_iris = iris.target

# print(X_iris)



k_means = cluster.KMeans(n_clusters=k)
k_means.fit(X_iris.reshape(-1, 1))

labels = k_means.labels_
centroids = k_means.cluster_centers_

for i in range(k):
    # select only data observations with cluster label == i
    ds = X_iris[np.where(labels==i)]
    # plot the data observations
    pyplot.plot(ds[:,0],ds[:,1],'o')
    # plot the centroids
    lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
    # make the centroid x's bigger
    pyplot.setp(lines,ms=15.0)
    pyplot.setp(lines,mew=2.0)
pyplot.show()
