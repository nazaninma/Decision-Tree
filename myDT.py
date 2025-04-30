from typing import List, Dict, Any
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

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
        # not neccessery
        # self.feature_importances_ = self._calculate_feature_importances()

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

        if ((depth>=self.max_depth and self.max_depth is not None )or (len(y)< self.min_samples_split)):
            class_num= np.bincount(y)
            lable= np.argmax(class_num)
            return {"class": lable}



        elif (len(np.unique(y))==1):
            return {"class": y[0]}
        
        else:
            index= self._best_split(X,y)['feature_idx']
            node={'feature_idx': index , 'children':{}}
            features= np.unique(X[:,index])
            for f in features:
                checking= np.where(X[:,index]==f)[0]
                my_x= X[checking]
                my_y=y[checking]


                my_x=np.delete(my_x,index,axis=1)
                node['children'][f]= self._build_tree(my_x, my_y,depth+1)




        # TODO: Implement the tree building logic
        # if (depth== self.max_depth) or () 
        # self._calculate_feature_importances
        # if depth== self.max_depth or len(X)< self.min_samples_split:
            # TODO
            # create new leaf node

            # return 

        # else:
        #     self._best_split(X,y)


        
        return 

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
        # bestFeature=""
        # important=0
        #
        # for attribute in X:
        #     important= max(important , _calculate_feature_importances(attribute))
        #     bestFeature= max

        # also can use gini
        entropy = self._calculate_entropy(y)
        feature_index=-1
        best = - float('inf')

        for i in range(X.shape[1]):
            feature_entropy= self._calculate_feature_entropy(X,y,i)
            if (best<entropy-feature_entropy):
                best= feature_entropy
                feature_index=i


        feature_dic={'feature_index': feature_index}
        return feature_dic


    def _calculate_feature_importances(self) -> np.ndarray:
        """
        Calculate feature importances based on the tree structure.

        Returns
        -------
        feature_importances : array-like of shape (n_features,)
            The feature importances.
        """
        # TODO: Implement feature importance calculation
        self.tree_
        pass

    def _calculate_entropy(self, y: np.ndarray) -> float:  #DONE
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

    def _calculate_gini(self, y: np.ndarray) -> float:   #DONE
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
        # TODO: Implement Gini index calculation
        pass
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
        # TODO: Implement feature entropy calculation
        feature_entropy=0
        # getting the feature using its index
        my_feature= X[:, feature_idx]
        # the number of the children when we use this specific feature for spliting
        children_num=np.unique(my_feature)

        for f in children_num:
            feature_children= np.where(my_feature==f)[0]
            label= y[feature_children]
            label_entropy=self._calculate_entropy(label)
            feature_entropy+= len(label)/len(y) * label_entropy
        
        # probabilities = class_counts[class_counts > 0] / len(y)
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

    def predict(self, X: np.ndarray) -> np.ndarray:  #DONE
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
        # TODO: Implement the prediction logic for a single sample
        pass



# a=2
# b=3
# c=a+b
# print('hello'+ c)
