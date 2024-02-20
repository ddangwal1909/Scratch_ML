from common_utils import *
from common_utils.model import Model

class LogisticRegression(Model):
    def __init__(self,model_name="LOGISTIC_REGRESSION",threshold=0.5):
        super().__init__(model_name=model_name)
        self.weights=None
        self.biases=None
        self.threshold=threshold
    
    def fit(self,x_train,y_train,learning_rate=0.001,iterations=100,show_verbose=False):
        dw,db=0,0
        num_samples,num_features = x_train.shape
        self.weights = np.zeros(num_features)
        self.biases = 0
        y_pred = np.zeros(num_samples)
        for iter in range(iterations):
            dw = (1/num_samples)*np.dot(x_train.T,y_pred - y_train)
            db = (1/num_samples)*np.sum(y_pred-y_train)
            self.weights = self.weights - learning_rate*dw
            self.biases = self.biases - learning_rate*db
            y_pred = self.sigmoid(np.dot(x_train,self.weights)+self.biases)
            accuracy = self.get_accuracy(y_pred,y_train)
            if show_verbose:
                print(f"ITERATION {iter+1} with learning rate:{learning_rate} --> accuracy:[{accuracy}], logloss:[{log_loss}]")
        
        return

    def sigmoid(self,linear_pred):
        return 1/(1+np.exp(-1*linear_pred))

    def get_accuracy(self,Y_pred,Y_train):
        assert Y_pred.size == Y_train.size
        matched = np.sum(Y_pred==Y_train)
        return matched/Y_pred.size

    def predict(self,x):
        linear_pred = np.dot(x,self.weights)+self.biases
        y_preds = self.sigmoid(linear_pred)
        y_preds = np.where(y_preds > self.threshold, 1, 0)
        return y_preds



