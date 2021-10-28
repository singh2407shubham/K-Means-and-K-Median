import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    """
    calculate euclidean distance 
    between two vectors/points
    used by the K-Means class
    """
    p1 = np.array(point1)
    p2 = np.array(point2)
    return np.sqrt(np.sum(np.square(p1 - p2)))
    
def manhattan_distance(point1, point2):
    """
    calculate taxicab distance
    between two given vectors
    used by the K-Medians class
    """
    p1 = np.array(point1)
    p2 = np.array(point2)
    return np.sum(np.abs(p1-p2))

def mean(arr):
    """
    calculate mean of an array
    called by K-Means
    """
    return np.mean(arr, axis=0)

def median(arr):
    """
    calculate median of an array
    called by K-Medians
    """
    return np.median(arr, axis=0)

def l2_normalise(samples):
    """
    calculate norm of a vector
    and divide the vector by the norm
    """
    normalised_samples = []
    for sample in samples:
        norm_l2 = np.linalg.norm(sample)
        sample = sample/norm_l2
        normalised_samples.append(sample)
    return normalised_samples


class KMeans:
  
    def __init__(self, K, max_iters=1000):
        # number of clusters
        self.K = K
        # iterations to be run for 
        # cluster optimisation
        self.max_iters = max_iters

        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def run(self, X):
        
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialise random indices from the sample as centroids
        # replace=False since we don't want two same indices
        rand_sample_indcs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in rand_sample_indcs]

        # cluster optimisation
        for _ in range(self.max_iters):
            # create clusters
            self.clusters = self.create_clusters(self.centroids)
            
            # evaluate new centroids 
            old_centroids = self.centroids
            self.centroids = self.get_centroids(self.clusters)

            # check if centroids have converged
            if self.if_converged(old_centroids, self.centroids):
                break
        
        return self.clusters
            
    def create_clusters(self, centroids):
        # re-initialise the clusters
        clusters = [[] for _ in range(self.K)]

        # find closest centroid for a given sample
        for idx, sample in enumerate(self.X):
            c_idx = self.closest_centroid(sample, centroids)
            clusters[c_idx].append(idx)
        return clusters

    def closest_centroid(self, sample, centroids):
        # evaluate closest centroid
        d = [self.distance(sample, point) for point in centroids]
        closest = np.argmin(d)
        return closest
 
    def distance(self, point1, point2):
        # calculate distance (euclidean in this case)
        d = euclidean_distance(point1, point2)
        return d
    
    def average(self, arr):
        # calculate mean
        return mean(arr)

    def get_centroids(self, clusters):
        
        centroids = np.zeros((self.K, self.n_features))
        
        # mean values of clusters --> centroids
        for idx, cluster in enumerate(clusters):
            # axis=0 --> mean from current cluster
            mean = self.average(self.X[cluster])
            centroids[idx] = mean
        return centroids

    def if_converged(self, old_centroids, new_centroids):
        # check if the centroids have converged
        d = [self.distance(old_centroids[idx], new_centroids[idx]) for idx in range(self.K)]
        return sum(d) == 0
    
    def b_cubed(self, clusters, categories):
        # method to evalute b-cubed measure
        # precision, recall and F-score
        sample_len = len(self.X)
        
        la = len(categories[0])
        lc = len(categories[1])
        lf = len(categories[2])
        lv = len(categories[3])
        
        p, r, f = [], [], []
        
        for cluster in clusters:
            a, b, c, d = 0, 0, 0, 0
            for obj in cluster:
                if obj in categories[0]:
                    a += 1
                elif obj in categories[1]:
                    b += 1
                elif obj in categories[2]:
                    c += 1
                elif obj in categories[3]:
                    d += 1
            
            l = len(cluster)
        
            p_ = ((a)**2 + (b)**2 + (c)**2 + (d)**2)/l
            p.append(p_)
            
            r_ = (a)**2/la + (b)**2/lc + (c)**2/lf + (d)**2/lv
            r.append(r_)
            
            f_ = 2*(a**2)/(la+l) + 2*(b**2)/(lc+l) + 2*(c**2)/(lf+l) + 2*(d**2)/(lv+l)
            f.append(f_)

        precision = sum(p)/sample_len
        recall = sum(r)/sample_len
        f_score = sum(f)/sample_len

        return precision, recall, f_score

    
class KMedians(KMeans):
    """
    inherited from K-Means
    keeping all methods same
    except the distance and averge
    methods are redefined
    """
    def distance(self, point1, point2):
        return manhattan_distance(point1, point2)

    def average(self, arr):
        return median(arr)

def file_handler(path):
        # files of different categories of objects
        files = ['animals', 'countries', 'fruits', 'veggies']
        categories = []
        data = []
        idx = 0
        for filename in files:
            # collect indeces for the given file
            # and put them into categories list
            idcs = []
            # input the correct path to the data directory
            f = open(path + filename, "r")
            for line in f:
                l = line.split()
                x = np.array(l[1:])
                x = x.astype(np.float64)
                data.append(x)
                idcs.append(idx)
                idx += 1
            categories.append(idcs)
        
        return data, categories

# test methods for the K-Means and K-Medians
# classes with and w/o L2 normalisation
def test_KMeans(k, data, categories):
    
    k = 9
    precision, recall, fscore = [], [] ,[]
    data = np.array(data)
    for num in range(k):
        
        km = KMeans(K=num+1)
        clus = km.run(data)
        p,r,f = km.b_cubed(clus, categories)
        
        precision.append(p)
        recall.append(r)
        fscore.append(f)

    X = np.arange(1,k+1,1)
    plt.xlabel("K")
    plt.title("K-Means B-Cubed Measure")
    plt.plot(X,precision,label="precision")
    plt.plot(X,recall,label="recall")
    plt.plot(X,fscore,label="fscore")
    plt.legend()
    plt.show()
    

def test_KMeans_l2_normalised(k, data, categories):
    
    precision, recall, fscore = [], [] ,[]
    data = l2_normalise(data)
    data = np.array(data)

    for num in range(k):
        
        km = KMeans(K=num+1)
        clus = km.run(data)
        p,r,f = km.b_cubed(clus, categories)
        
        precision.append(p)
        recall.append(r)
        fscore.append(f)

    X = np.arange(1,k+1,1)
    plt.xlabel("K")
    plt.title("K-Means B-Cubed Measure (L2 norm)")
    plt.plot(X,precision,label="precision")
    plt.plot(X,recall,label="recall")
    plt.plot(X,fscore,label="fscore")
    plt.legend()
    plt.show()

def test_KMedians(k, data, categories):
    
    precision, recall, fscore = [], [] ,[]
    data = np.array(data)
    for num in range(k):
        
        km = KMedians(K=num+1)
        clus = km.run(data)
        p,r,f = km.b_cubed(clus, categories)
        
        precision.append(p)
        recall.append(r)
        fscore.append(f)

    X = np.arange(1,k+1,1)
    plt.xlabel("K")
    plt.title("K-Medians B-Cubed Measure")
    plt.plot(X,precision,label="precision")
    plt.plot(X,recall,label="recall")
    plt.plot(X,fscore,label="fscore")
    plt.legend()
    plt.show()

def test_KMedians_l2_normalised(k, data, categories):
    
    precision, recall, fscore = [], [] ,[]
    data = np.array(data)
    for num in range(k):
        
        km = KMedians(K=num+1)
        clus = km.run(data)
        p,r,f = km.b_cubed(clus, categories)
        
        precision.append(p)
        recall.append(r)
        fscore.append(f)

    X = np.arange(1,k+1,1)
    plt.xlabel("K")
    plt.title("K-Medians B-Cubed Measure (L2 norm)")
    plt.plot(X,precision,label="precision")
    plt.plot(X,recall,label="recall")
    plt.plot(X,fscore,label="fscore")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # do not change anything above this line

    # enter the location of the file (path)
    path = "CA2data/"
    data, categories = file_handler(path)

    # testing
    # k value (9 as per our case)
    k = 9
    # un-comment appropriate function below to plot the b-cubed measure data 
    
    test_KMeans(k, data, categories)
    # test_KMeans_l2_normalised(k, data, categories)
    # test_KMedians(k, data, categories)
    # test_KMedians_l2_normalised(k, data, categories)
