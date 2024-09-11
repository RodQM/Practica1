# Practica1


import pandas as pd
from ydata_profiling import ProfileReport


df = pd.read_csv("/Users/rodrigo/Desktop/ESCUELA/MACHINE LEARNING/bezdekIris.csv", header=1)


profile = ProfileReport(df, title="Reporte del Dataset Iris", explorative=True)
profile.to_file("reporte_iris.html")

print("Reporte generado y guardado como 'reporte_iris.html'.")


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


print(df.columns)


X = df[['S1', 'S2', 'P1', 'P2']]


kmeans = KMeans(n_clusters=3)
kmeans.fit(X)


df['Cluster'] = kmeans.labels_


plt.figure(figsize=(10,6))
sns.scatterplot(x='Sepal_Lenght', y='Petal_Width', hue='Cluster', data=df, palette='deep')
plt.title("Clusters formados por K-Means en el dataset Iris")
plt.show()


centroids = kmeans.cluster_centers_
print("Centroides de los clusters:\n", centroids)
