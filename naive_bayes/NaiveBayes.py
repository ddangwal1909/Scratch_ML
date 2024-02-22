from common_utils.model import Model
from common_utils import *

class NaiveBayes(Model):
    def __init__(self,model_name="Naive Bayes"):
        super().__init__(model_name)
        self.x_train = None
        self.y_train = None
        self.num_classes= None
        ### prior probability
        self.p_y = None
        self.y_mean = None
        self.y_std = None

    def _get_prior_probability(self,y):
        p_y = np.bincount(y)/len(y)
        #print(p_y)
        return p_y
    
    def _get_mean_class(self,x,y):
        num_samples,num_features = x.shape
        num_class = max(np.unique(y))
        mean_classes = np.zeros((num_class+1,num_features),dtype=np.float64)
        for curr_label in range(num_class+1):
            curr_x  = x[y==curr_label,:]
            mean_classes[curr_label] = np.mean(curr_x,axis=0)
        return mean_classes


    
    def _get_standard_deviation_class(self,x,y):
        num_samples,num_features = x.shape
        num_class = max(np.unique(y))
        std_classes = np.zeros((num_class+1,num_features),dtype=np.float64)
        for curr_label in range(num_class+1):
            curr_x  = x[y==curr_label,:]
            std_classes[curr_label] = np.std(curr_x,axis=0)
        return std_classes

    def _get_gaussian_probability(self,x,class_y):
        curr_mean = self.y_mean[class_y,:]
        curr_std = self.y_std[class_y,:]
       # print(curr_mean,curr_std)
        gaussian_prob = np.array(1/(math.sqrt(2*math.pi)*curr_std))*np.exp(-1*((x-curr_mean)**2)/(2*(curr_std**2)))
        #print(x.shape,gaussian_prob.shape)
        return gaussian_prob

    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.num_classes = np.max(np.unique(y_train))
        self.p_y = self._get_prior_probability(y_train)
        self.y_mean = self._get_mean_class(x_train,y_train)
        self.y_std = self._get_standard_deviation_class(x_train,y_train)



    def _calculate_posterior_prob(self,class_y,x):
        probabilities = np.concatenate((self._get_gaussian_probability(x,class_y),np.array([self.p_y[class_y]])))
        return np.sum(np.log(probabilities))

    def predict(self,x_test):
        #### for each y_possible calculate the probability
        predictions =[]
        for x in x_test:
            curr_x = []
            for y in range(self.num_classes+1):
                curr_x.append(self._calculate_posterior_prob(y,x))
            predictions.append(np.argmax(curr_x))
        
        return np.array(predictions)

