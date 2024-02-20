from common_utils.model import Model
from common_utils import *
class KNN(Model):
    def __init__(self,model_name="KNN",k_neighbors=2):
        super().__init__(model_name)
        self.k_neighbors = k_neighbors
        self.x_train=None
        self.y_train=None

    def _get_euclidean_distance(self,x):
        distances = [np.sqrt(np.sum((x_t-x)**2)) for x_t in self.x_train]
        return distances


    def fit(self,x_train,y_train):
        self.x_train=x_train
        self.y_train=y_train
    
    def predict(self,x_test):

        ### find euclidean distances for each x point
        distances = [self._get_euclidean_distance(x) for x in x_test]
        
        ### find closest k_neighbours for each x_test point
        k_nearest_ys = [np.argsort(_)[:self.k_neighbors] for _ in distances]
        
        ### get maximum vote for each x_test
        k_nearest_vote_prediction = np.array([Counter([self.y_train[y_idx] for y_idx in _]).most_common(1)[0][0] for _ in k_nearest_ys])

        #print(k_nearest_vote_prediction)
        return k_nearest_vote_prediction
        
