from common_utils import *
from common_utils.model import Model
class LinearRegression(Model):
    def __init__(self,model_name="LINEAR REGRESSION"):
        super().__init__(model_name)
        self.weights = None
        self.biases = None
        self.mse_train = None

    def fit(self,X_train,Y_train,iterations=10,learning_rate=0.001,show_verbose=False,early_stop=False):
        num_samples,num_features = X_train.shape
        self.weights = np.zeros(num_features)
        self.biases = 0
        ## gradients
        db,dw=0,0
        ### initialize y_pred
        Y_pred = self.predict(X_train)
        self.mse_train={}
        max_so_far=None
        ## gradient descent
        for iter in range(iterations):
            dw = (1/num_samples)*np.dot(X_train.T,Y_pred-Y_train)
            db = (1/num_samples)*np.sum(Y_pred-Y_train)
            self.weights = self.weights - learning_rate*dw
            self.biases = self.biases - learning_rate*db
            self.mse_train[iter+1]=self.mse(Y_train,self.predict(X_train))
            if show_verbose:
                print(f"ITERATION {iter+1} with learning rate:{learning_rate} --> mse:[{self.mse_train[iter+1]}]")
            if early_stop:
                if max_so_far is not None:
                    if self.mse_train[iter+1]>max_so_far:
                        print(f"EARLY STOPPED AT Iteration: {iter+1}!")
                        break
            if max_so_far is None:
                max_so_far=self.mse_train[iter+1]
            else:
                max_so_far=max(max_so_far,self.mse_train[iter+1])
        return
        
    
    def predict(self,X):
        Y_pred = np.dot(X,self.weights.T) + self.biases
        return Y_pred
    

