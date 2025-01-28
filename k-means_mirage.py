import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans

def load_embeddings():
   
    mirage_raw = pd.read_csv("clustering_3.csv")

 
    row, col = mirage_raw.shape
    print(f'There are {row} rows and {col} columns')
    print(mirage_raw.head(10))

    
    raw_scaled = mirage_raw.copy()




    # Scaling the data to keep the different attributes in same range.
    raw_scaled[raw_scaled.columns] = StandardScaler().fit_transform(raw_scaled)
    print(raw_scaled.describe())

    return raw_scaled


def pca_embeddings(df_scaled):

    pca_2 = PCA(n_components=2)
    pca_2_result = pca_2.fit_transform(df_scaled)
    print('Explained variation per principal component: {}'.format(pca_2.explained_variance_ratio_))
    print('Cumulative variance explained by 2 principal components: {:.2%}'.format(
        np.sum(pca_2.explained_variance_ratio_)))

    # Results from pca.components_
    dataset_pca = pd.DataFrame(abs(pca_2.components_), columns=df_scaled.columns, index=['PC_1', 'PC_2'])
    print('\n\n', dataset_pca)
    dataset_pca.to_csv('pca_result.csv')

    print("\n*************** Most important features *************************")
    print('As per PC 1:\n', (dataset_pca[dataset_pca > 0.3].iloc[0]).dropna())
    print('\n\nAs per PC 2:\n', (dataset_pca[dataset_pca > 0.3].iloc[1]).dropna())
    print("\n******************************************************************")

    return pca_2_result, pca_2

def kmean_hyper_param_tuning(data):

    # candidate values for our number of cluster
    parameters = [2, 3, 4, 5, 10, 15, 20, 25]

    # instantiating ParameterGrid, pass number of clusters as input
    parameter_grid = ParameterGrid({'n_clusters': parameters})

    best_score = -1
    kmeans_model = KMeans()     # instantiating KMeans model
    silhouette_scores = []

    # evaluation based on silhouette_score
    for p in parameter_grid:
        kmeans_model.set_params(**p)    # set current hyper parameter
        kmeans_model.fit(data)          # fit model on wine dataset, this will find clusters based on parameter p

        ss = metrics.silhouette_score(data, kmeans_model.labels_)   # calculate silhouette_score
        silhouette_scores += [ss]       # store all the scores

        print('Parameter:', p, 'Score', ss)

        # check p which has the best score
        if ss > best_score:
            best_score = ss
            best_grid = p

    # plotting silhouette score
    plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='#722f59', width=0.5)
    plt.xticks(range(len(silhouette_scores)), list(parameters))
    plt.title('Silhouette Score', fontweight='bold')
    plt.xlabel('Number of Clusters')
    plt.show()

    return best_grid['n_clusters']

def visualizing_results(pca_result, label, centroids_pca):

    # ------------------ Using Matplotlib for plotting-----------------------
    x = pca_result[:, 0]
    y = pca_result[:, 1]
    color = np.where(label == 1, "red", "blue")

    plt.scatter(x, y, c=color, alpha=0.5, s=200)  # plot different colors per cluster
    plt.title('Performance Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, linewidths=1.5,
                color='red', edgecolors="black")

    plt.show()
    print(label)


def main():
    print("1. Loading Wine dataset\n")
    data_scaled = load_embeddings()

    print("\n\n2. Reducing via PCA\n")
    pca_result, pca_2 = pca_embeddings(data_scaled)

    print("\n\n3. HyperTuning the Parameter for KMeans\n")
    optimum_num_clusters = kmean_hyper_param_tuning(data_scaled)
    print("optimum num of clusters =", optimum_num_clusters)

    # fitting KMeans
    kmeans = KMeans(n_clusters=optimum_num_clusters)
    kmeans.fit(data_scaled)
    centroids = kmeans.cluster_centers_
    centroids_pca = pca_2.transform(centroids)

    print("\n\n4. Visualizing the data")
    visualizing_results(pca_result, kmeans.labels_, centroids_pca)
    print(kmeans.labels_)
    print(pca_result)
    print(centroids_pca)



if __name__ == "__main__":
    main()

