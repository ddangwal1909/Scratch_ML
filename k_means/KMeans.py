from common_utils.model import Model
from common_utils import *

class KMeans(Model):
    def __init__(self,model_name="K-Means"):
        super().__init__(model_name)
        self.centroids = None
        self.num_clusters = None
        self.sample_cluster=None
        self.x_train = None
    
    def _get_euclidean_distance(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def _get_wcss(self,cluster_samples,centroid):
        wcss = np.sum((cluster_samples-centroid)**2)
        return wcss

    def _get_closest_centroid(self,x_sample):
        min_distance_centroid = np.sqrt(np.sum((self.centroids-x_sample)**2,axis=1)).argmin()
        return min_distance_centroid

    def fit(self,x_train,num_clusters=5,num_iterations=1000,show_plot=False):
        num_samples,num_features = x_train.shape
        self.x_train =x_train
        self.sample_cluster = np.zeros(num_samples)
        self.centroids = np.random.rand(num_clusters,num_features)
        self.num_clusters=num_clusters
        for iter in range(num_iterations):
            

            ## get closest centroid
            for idx,x_idx in enumerate(x_train):
                self.sample_cluster[idx]=self._get_closest_centroid(x_idx)
            
            if show_plot:
                self.plot()
            
            ## calculate WCSS
            wcss=0
            for cluster in range(num_clusters):
                idx_samples_curr_cluster = np.argwhere(self.sample_cluster==cluster).flatten()
                wcss+=self._get_wcss(x_train[idx_samples_curr_cluster,:],self.centroids[cluster])
            print(f"After iteration {iter} for {num_clusters} clusters : WCSS-->{wcss}, {self.centroids}, {self.sample_cluster}")

            ## get new centroid based on mean of current clusters
            for cluster in range(num_clusters):
                idx_samples_curr_cluster = np.argwhere(self.sample_cluster==cluster).flatten()
                curr_sample_x_train = x_train[idx_samples_curr_cluster,:]
                new_centroid = np.mean(curr_sample_x_train,axis=0)
                self.centroids[cluster]= self.centroids[cluster] if np.any(np.isnan(new_centroid)) else new_centroid
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for cluster in range(self.num_clusters):
            idx_samples_curr_cluster = np.argwhere(self.sample_cluster==cluster).flatten()
            point = self.x_train[idx_samples_curr_cluster].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()



    def predict(self,x_test):
        predictions = []
        for x_sample in x_test:
            predictions.append(self._get_closest_centroid(x_sample))
        
        return np.array(predictions)
