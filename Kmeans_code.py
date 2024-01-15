# -*- coding: utf-8 -*-
"""Iris Dataset Kmeans.ipynb

Este codigo esta diseñado para aplicar la técnica de ML Kmeans sobre el iris dataset. 
Se van a utilizar las columnas de sepal length sepal width petal length y petal width solamente para utilizar esta técnica de Unsupervised Learning. 
Luego de encontrar la cantidad correcta de cluster se va a comparar con lo que se obtiene de los datos de 'iris.csv'
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("/content/iris.csv")
df.head()


from sklearn.cluster import KMeans
X = df.iloc[:,0:4] # defino X como un dataset sin etiquetas ( sin la columna 'variety')

# Observo el valor de wcss para una cantidad variable de clusters, luego gráfico para observar cuál es la cantidad adecuada
wcss = []       
for i in range (1,11):
  kmeans = KMeans(n_clusters=i ,
                  init = "k-means++",
                  max_iter = 300,
                  n_init= 10 ,
                  random_state=0)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title("Método del codo")
plt.xlabel("Número de Clusters")
plt.ylabel("WCSS(k)")
plt.xticks(np.arange(1, 11, step=1)) # para tener el número de clusters
plt.show()

# Utilizo 3 cluster debido a los resultados anteriores.

kmeans = KMeans( n_clusters = 3 , init = "k-means++", max_iter = 300, n_init =10, random_state=0 )
y_kmeans = kmeans.fit_predict(X)

# Creo el dataframe Rkmeans para difernciar los resultados de df
Rkmeans = X
Rkmeans['Cluster'] = y_kmeans
Rkmeans.head()

### Creo un subplot donde el gráfico e la izquierda es un scatterplot del 'petal.width' en función de 'petal.length' coloreado (hue) segun 'variety' features del dataframe df
### el gráfico de la derecha corresponde también a un scatterplot de las mismas features pero ahora el coloreado(hue) es según 'Cluster' del dataframe Rkmeans.

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
sns.scatterplot(df , x= 'petal.length', y= 'petal.width', hue='variety' , ax=axes[0])
axes[0].set_title('Gráfico con etiquetas IrisDataset')

sns.scatterplot(Rkmeans , x = 'petal.length', y='petal.width', hue='Cluster', palette = 'Set2', ax = axes[1])
plt.scatter(kmeans.cluster_centers_[:,2], kmeans.cluster_centers_[:,3],
            s = 300, c = "Yellow", label ="Baricentros")
axes[1].set_title('Gráfico Utilizando KMeans')


plt.tight_layout()
plt.show()
