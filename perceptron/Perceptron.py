from common_utils.model import Model
from common_utils import *

class Perceptron(Model):
    def __init__(self,model_name="Perceptron"):
        super().__init__(model_name)
        self.activation_function = None
        self.weights = None
        self.bias = None

    def _activation_function_map(self,activation_function):
        mapping = {
            "step":self.step,
            "ReLU":self.relu,
            "sigmoid":self.sigmoid,
        }
        return mapping[activation_function]
    

    def relu(self,x):
        pass
    def sigmoid(self,x):
        pass
    def step(self,x):
        x = np.array(x)
        x = np.where(x>0,1,0)
        return x

    def fit(self,x_train,y_train,learning_rate=0.01,activation_function="step",num_iterations=10000):
        num_samples,num_features = x_train.shape
        self.activation_function = self._activation_function_map(activation_function)
        self.weights = np.zeros(num_features)
        self.bias = 0
        for iteration in range(num_iterations):
            for sample in range(num_samples):
                prediction = self.activation_function(np.dot(self.weights.T,x_train[sample,:])+self.bias)
                dw = learning_rate*(y_train[sample] - prediction)*x_train[sample,:]
                db = learning_rate*(y_train[sample] - prediction)
                self.weights = self.weights + dw
                self.bias = self.bias + db
            #print(self.weights,self.bias)

    def predict(self,x_test):
        preds = self.activation_function(np.dot(x_test,self.weights)+self.bias)
        return preds
