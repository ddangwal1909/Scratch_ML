from linear_regression.LinearRegression import LinearRegression
from common_utils import *
class LassoRegression(LinearRegression):
    def __init__(self,model_name="Lasso regression"):
        super().__init__(model_name)

    def fit(self,X_train,Y_train,iterations=10,learning_rate=0.001,penalty_coefficient = 0.001,show_verbose=False):
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
            dw = (1/num_samples)*np.dot(X_train.T,Y_pred-Y_train) + penalty_coefficient
            db = (1/num_samples)*np.sum(Y_pred-Y_train)
            self.weights = self.weights - learning_rate*dw
            self.biases = self.biases - learning_rate*db
            self.mse_train[iter+1]=self.mse(Y_train,self.predict(X_train))
            if show_verbose:
                print(f"ITERATION {iter+1} with learning rate:{learning_rate} --> mse:[{self.mse_train[iter+1]}]")
        return