import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.linalg import eigs
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.csgraph import laplacian

        
class Kmeans:
    def __init__(self, nclusters=2, random_state=47, tolerance=0.1, iterations=10, init_method='random'):
        """
        Initializes a new instance of KMeans.
        :param nclusters: number of clusters. Defaults to 2. Type: int
        :param random_state: random_state used to select centroid at random for kmeans. Defaults to 47. Type: int
        :param tolerance: criteria to stop when new centroid changes by more than this value. Defaults to 0.1. Type: float
        :param iterations: Number of iterations before stopping. Defaults to 10. Type: int
        :param init_method: 'random' implies kmeans or random initialization, 'kpp' implies 'kmeans++' method of centroid initialization. Defaults to 'random'. Type: str
        """
        self.nclusters=nclusters
        self.random_state=random_state
        self.tolerance=tolerance
        self.iterations=iterations
        self.final_centroids=None
        self.init_method=init_method
    
    def init_centroids(self, data):
        len_data=len(data)
        r = np.random.RandomState(self.random_state)
        rand_idx=r.choice(len_data, self.nclusters)
        centroids=data[rand_idx]
        return centroids

    def init_centroids_kpp(self, data):
        len_data=len(data)
        idxs=[]
        r = np.random.RandomState(self.random_state)
        rand_idx=r.choice(len_data, 1)
        c1=data[rand_idx]
        idxs.append(rand_idx[0])
        for c in range(self.nclusters-1):
            dist=self.calc_distance(data,c1)
            if c==0:
                dists=dist.reshape((dist.shape[0],1))
                rand_idx=np.argmax(dists)
            else:
                dists=np.append(dists,dist.reshape(dist.shape[0],1),axis=1)
                rand_idx=np.argmax(np.min(dists,axis=1))
            c1=data[rand_idx]
            idxs.append(rand_idx)
        centroids=data[idxs]
        return centroids

    def calc_distance(self, X, Y):
        dist=np.sqrt(np.sum((X - Y)**2,axis=1))
        return dist

    def create_clusters(self, data, centroids):
        cluster=np.zeros((data.shape[0],))
        for pt_idx, point in enumerate(data):
            mini,minv=0,np.inf
            for idx, centroid in enumerate(centroids):
                dist=np.sqrt(np.sum((point - centroid)**2))
                if dist<minv:
                    minv=dist
                    mini=idx
            cluster[pt_idx]=mini
        return cluster

    def calculate_centroids(self, data, clusters):
        centroids=np.zeros((self.nclusters,data.shape[1]))
        for centroid in range(self.nclusters):
            d_c=data[np.where(clusters==centroid)]
            centroids[centroid]=np.mean(d_c,axis=0)
        return centroids

    def fit_predict(self, data):
        data = np.array(data)
        if self.init_method=='kpp':
            prev_centroids = self.init_centroids_kpp(data)
        else:
            prev_centroids = self.init_centroids(data)
        for itr in range(self.iterations):
            clusters=self.create_clusters(data, prev_centroids)
            centroids=self.calculate_centroids(data,clusters)
            dists=self.calc_distance(prev_centroids,centroids)
            high_dist=np.where(dists>self.tolerance)
            # display(X, centroids, clusters)
            if high_dist[0].any():
                prev_centroids=centroids
            else:
                break
        self.final_centroids=centroids
        return clusters

    def return_centroids(self, data):
        data = np.array(data)
        return self.init_centroids(data), self.final_centroids


class SpecCluster:
    def __init__(self, nclusters=2, nneighbors=10, gamma=2.0, var=1, affinity='rbf'):
        self.nclusters = nclusters
        self.nneighbors = nneighbors
        self.gamma = gamma
        self.var = var
        self.affinity = affinity
    
    def fit(self, X):
        ### Compute affinity matrix based on the affinity method
        if self.affinity == 'nearest_neighbors':
            W = kneighbors_graph(X, n_neighbors=self.nneighbors, mode='connectivity')
            W = 0.5 * (W + W.T)
        elif self.affinity == 'rbf' :
            W = self.var * rbf_kernel(X, gamma = self.gamma)
        ### Compute diagonal matrix of affinity matrix
        D = np.diag(np.ravel(np.sum(W, axis=1)))
        ### Calculate laplacian matrix
        L = D - W
        ### Calculate eigen values or vectors of the affinity matrix
        eigvals, eigvecs = eigs(L, k=self.nclusters, which='SM')
        eigvecs = eigvecs[:, 1:]
        eigvecs = np.real(eigvecs)
        ### Perform kmeans clustering on eigen vectors
        kmean = Kmeans(nclusters=self.nclusters, init_method='kpp')
        self.labels_ = kmean.fit_predict(eigvecs)
        self.cluster_centers_ = kmean.return_centroids(eigvecs)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    