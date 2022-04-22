from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
X = iris['data']
y = iris['target']

#Створення об'єкту КМеаns
kmeans = KMeans(n_clusters=8, init='k-means++', n_init=10,
                max_iter=300, tol=0.0001,
                verbose=0, random_state=None,
                copy_x=True, algorithm='auto')

#Навчання моделі кластеризації КМеаns
kmeans.fit(X)

#Передбачення вихідних міток для всіх точок сітки
y_kmeans = kmeans.predict(X)

#Графічне відображення областей та виділення їх кольором
plt.figure()
plt.scatter(X[:,0], X[:,1], c=y_kmeans, s=50, cmap='viridis')
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:,0], cluster_centers[:,1],
            s=200, c='black', alpha=0.5)


#Визначаємо функцію для пошуку кластерів
def find_clusters(X, n_clusters, rseed=2):
    #Вибір кластерів рандомно
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        #Визначення міток на основі найближчих центрів
        labels = pairwise_distances_argmin(X, centers)
        #Пошук нових центрів за допомогою точок
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        #Перевірка на збіжність
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels


centers, labels = find_clusters(X, 3)
plt.figure()
plt.scatter(X[:,0], X[:,1], c=labels, s=50, cmap='viridis')

#Пошук кластерів при rseed=0
centers, labels = find_clusters(X, 3, rseed=0)
plt.figure()
plt.scatter(X[:,0], X[:,1], c=labels, s=50, cmap='viridis')

labels = KMeans(3, random_state=0).fit_predict(X)
plt.figure()
plt.scatter(X[:,0], X[:,1], c=labels, s=50, cmap='viridis')
plt.show()