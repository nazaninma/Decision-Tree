from typing import List, Dict, Any
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import math


class MultiNodeCategoricalDecisionTree(BaseEstimator, ClassifierMixin):
    """
    A multi-node categorical decision tree classifier.
    
    This classifier is designed to work with categorical features and can have
    multiple branches at each node, unlike binary decision trees.
    
    Parameters
    ----------
    max_depth : int, optional (default=None)
        The maximum depth of the tree. If None, the tree will expand until all
        leaves are pure or until all leaves contain less than min_samples_split samples.
    
    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.
    
    Attributes
    ----------
    tree_ : dict  
        The tree structure stored as a nested dictionary.
    
    n_classes_ : int
        The number of classes.
    
    classes_ : array-like of shape (n_classes,)
        The class labels.
    
    n_features_ : int
        The number of features when `fit` is performed.
    
    feature_importances_ : array-like of shape (n_features,)
        The feature importances based on the amount of criterion reduction achieved. 
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiNodeCategoricalDecisionTree':
        """
        Build a multi-node categorical decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]

        # Build the tree
        self.tree_ = self._build_tree(X, y)

        # Calculate feature importances
        self.feature_importances_ = self._calculate_feature_importances()

        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Dict[str, Any]:
        """
        Recursively build the decision tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        depth : int
            The current depth of the tree.

        Returns
        -------
        node : dict
            A dictionary representing the current node in the tree.
        """
        #if all in class are same => label that class
        if(len(X)==0):
            return None
        if(len(np.unique(y))==1):
            return {"class":y[0]}
        # if we are in max depth or we don't have enough sample

        if((self.max_depth is not None and depth>=self.max_depth) or (len(y)<self.min_samples_split)):
            num=np.bincount(y)
            lable=np.argmax(num)
            return {"class" : lable}
        
        #end of break part


        current=self._best_split(X,y)
        idx=current['feature_idx']
        if(idx==-1):
            num=np.bincount(y)
            lable=np.argmax(num)
            return {"class" : lable}
        
        node={'feature_idx' : idx,'children' : {},'majority_class': np.bincount(y).argmax()}
        
        feature_val=np.unique(X[:,idx])

        for v in feature_val:
            indic=np.where(X[:,idx]==v)[0]
            new_x=X[indic]
            new_y=y[indic]

            #remove the feature that no children use it again
            new_x=np.delete(new_x,idx,axis=1)
            if len(new_y) > 0:
                node["children"][v]=self._build_tree(new_x,new_y,depth+1)
            else:
                label = np.argmax(np.bincount(y))
                node["children"][v] = {"class": label}
        return node
        

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Find the best split for a node.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        best_split : dict
            A dictionary containing information about the best split.
        """
        # TODO: Implement the best split selection logic

        parent_gini=self._calculate_gini(y)

        feature_index=-1
        best=-float('inf')
        ans_val=None

        num=X.shape[1]

        for i in range(num):
            ent=self._calculate_feature_gini(X,y,i)
            feature_gini = parent_gini- ent
            if(best<feature_gini):
                best=feature_gini
                feature_index=i
        
        dict= {'feature_idx': feature_index}
        return dict

    def _calculate_feature_importances(self) -> np.ndarray:  #not for training part for calculate the final result
        """
        Calculate feature importances based on the tree structure.

        Returns
        -------
        feature_importances : array-like of shape (n_features,)
            The feature importances.
        """
        # TODO: Implement feature importance calculation
        
        pass

    def _calculate_entropy(self, y: np.ndarray) -> float:   
        """
        Calculate the entropy of a dataset.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        entropy : float
            The calculated entropy.
        """
        # TODO: Implement entropy calculation
        class_counts = np.bincount(y)
        entropy=0
        possibility=0
        for i in class_counts:
            if (i>0):
                possibility=i / len(y)
                entropy-=( possibility * np.log2(possibility))
        
        # probabilities = class_counts[class_counts > 0] / len(y)
        return entropy

    def _calculate_gini(self, y: np.ndarray) -> float:
        """
        Calculate the Gini index of a dataset.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        gini : float
            The calculated Gini index.
        """
        class_counts = np.bincount(y)
        gini=1
        possibility=0
        for i in class_counts:
            if (i>0):
                possibility=i / len(y)
                gini-= (possibility * possibility)
        
        # probabilities = class_counts[class_counts > 0] / len(y)
        return gini

    def _calculate_feature_entropy(self, X: np.ndarray, y: np.ndarray, feature_idx: int) -> float:
        """
        Calculate the entropy of a specific feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        feature_idx : int
            The index of the feature to calculate entropy for.

        Returns
        -------
        feature_entropy : float
            The calculated feature entropy.
        """
        feature_entropy=0.

        value=X[:,feature_idx]
        child=np.unique(value)
        

        for v in child:
            child_num=np.where(value==v)[0]
            child_feature_entropy=y[child_num]
            ent=self._calculate_entropy(child_feature_entropy)

            w=len(child_feature_entropy)/len(y)
            feature_entropy+= w*ent
        
        return feature_entropy
       

    def _calculate_feature_gini(self, X: np.ndarray, y: np.ndarray, feature_idx: int) -> float:
        """
        Calculate the Gini index of a specific feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        feature_idx : int
            The index of the feature to calculate Gini index for.

        Returns
        -------
        feature_gini : float
            The calculated feature Gini index.
        """
        # TODO: Implement feature Gini index calculation
        feature_gini=0
        # getting the feature using its index
        my_feature= X[:, feature_idx]
        # the number of the children when we use this specific feature for spliting
        children_num=np.unique(my_feature)

        for f in children_num:
            feature_children= np.where(my_feature==f)[0]
            label= y[feature_children]
            label_gini=self._calculate_gini(label)
            feature_gini+= len(label)/len(y) * label_gini
        
        # probabilities = class_counts[class_counts > 0] / len(y)
        return feature_gini

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self)
        X = check_array(X)
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x: np.ndarray) -> Any:
        """
        Predict class for a single sample.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            The input sample.

        Returns
        -------
        y : Any
            The predicted class.
        """
        root = self.tree_  

        # Traverse the tree until a leaf root is reached
        while "class" not in root:

            
            # Get the value of the feature for this sample
            feature_value = x[root["feature_idx"]]
            
            # Traverse to the child root based on the feature value
            if feature_value in root["children"]:
                root = root["children"][feature_value]
            else:
                return root['default_class']

        # Once we reach a leaf, return the class
        return root["class"]

