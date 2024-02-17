from common_utils import *
class Model:
    def __init__(self,model_name:str=None):
        self.model_name=None
    
    def fit(self,X_train,Y_train):
        pass

    def predict(self,X):
        pass

    def get_evaluation_metrics(self,type_eval:str=None):
        if type_eval is None:
            raise Exception("Please provide a type_eval")
    
    def mse(self,Y_train,Y_pred):
        num_samples = Y_train.shape[0]
        mse = (1/num_samples)*(np.sum((Y_pred-Y_train)**2))
        return mse
