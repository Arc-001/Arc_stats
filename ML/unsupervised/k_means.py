import numpy as np

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): (K, n) centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    
    """
    print (len(X))
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    
    def l2sq(x,mu):
        return (x[0]-mu[0])**2 + (x[1]-mu[1])**2
    for index, cord in enumerate(X):
        cent_dist = []
        for mu in centroids:
            cent_dist.append(l2sq(cord,mu))
        
        idx[index]=np.argmin(cent_dist)
    
    return idx

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    m, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        points = np.array([X[j] for j in range(len(idx)) if idx[j] == k])
        centroids[k] = np.mean(points, axis  = 0)
    return centroids