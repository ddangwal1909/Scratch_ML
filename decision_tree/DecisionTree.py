from common_utils.model import Model
from common_utils import *
from collections import Counter
class DecisionTreeNode:
    def __init__(self,value=None,left=None,right=None,best_feature=None,best_threshold=None):
        self.value = value
        self.left=left
        self.right=right
        self.best_feature=best_feature
        self.best_threshold = best_threshold


class DecisionTree(Model):
    def __init__(self,min_samples_split=2, max_depth=10, n_features=None):
        self.root=None
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
    

    def _most_frequent_label(self,Y):
        counter = Counter(Y)
        value = counter.most_common(1)[0][0]
        return value
    
    def _get_entropy(self,Y):
        freq = np.bincount(Y)
        probabilities = freq/len(Y)
        entropy = -1*np.sum([p*np.log(p) for p in probabilities if p>0])
        return entropy

    def _split(self,X_Column,threshold):
        left_idx = np.argwhere(X_Column<=threshold).flatten()
        right_idx = np.argwhere(X_Column>threshold).flatten()
        return left_idx,right_idx


    def _get_information_gain(self,X,Y,threshold):
        
        #### get entropy of parent
        parent_entropy = self._get_entropy(Y)

        #### split based on current threshold
        left_idx,right_idx = self._split(X,threshold)

        if len(left_idx)==0 or len(right_idx)==0:
            return 0

        ## calculate information gain
        e_l = self._get_entropy(Y[left_idx])
        e_r = self._get_entropy(Y[right_idx])
        total_child_entropy = (len(left_idx)/len(Y))*e_l + (len(right_idx)/len(Y))*e_r
        
        return parent_entropy-total_child_entropy

    def _get_best_split(self,X,Y,selected_idxs):

        best_gain = -1
        best_split_idx,best_split_threshold = None,None
        
        for feat in selected_idxs:
            X_curr = X[:,feat]
            thresholds = np.unique(X_curr)
            for curr_threshold in thresholds:
                curr_gain = self._get_information_gain(X_curr,Y,curr_threshold)
                if curr_gain>best_gain:
                    best_split_idx=feat
                    best_split_threshold=curr_threshold
                    best_gain=curr_gain
        return best_split_idx,best_split_threshold
    



    def _build_tree(self,X,Y,curr_depth=0):
        # print(X,Y)
        num_samples,num_features = X.shape
        num_labels = len(np.unique(Y))

        ### check stop condition
        if (num_labels==1) or (num_samples<self.min_samples_split) or (curr_depth>self.max_depth):
            ## create the leaf node
            # print(Y)
            leaf = DecisionTreeNode(value=self._most_frequent_label(Y))
            return leaf
        
        ### if not any stop condition; then try to further build

        ## select random features from all features
        selected_features = np.random.choice(num_features,self.n_features,replace=False)

        #### get best split for current node based on selected_features
        best_feature,best_split_threshold=self._get_best_split(X,Y,selected_features)

        ### create children
        left_idx,right_idx = self._split(X[:,best_feature],best_split_threshold)
        left = self._build_tree(X[left_idx,:],Y[left_idx],curr_depth+1)
        right = self._build_tree(X[right_idx,:],Y[right_idx],curr_depth+1)
        return DecisionTreeNode(left=left,right=right,best_feature=best_feature,best_threshold=best_split_threshold)
        

    def fit(self,X,Y):
        ### number of samples and features
        num_samples,num_features = X.shape
        self.n_features = num_features
        self.root = self._build_tree(X,Y)

    def _traverse_decision_tree(self,root,x):
        if root.value is not None:
            return root.value
        
        if x[root.best_feature]<=root.best_threshold:
            return self._traverse_decision_tree(root.left,x)
        return self._traverse_decision_tree(root.right,x)


    def predict(self,X):
        results = np.array([self._traverse_decision_tree(self.root,x) for x in X])
        return results
    
