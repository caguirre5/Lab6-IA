'''
Universidad del Valle de Guatemala
Inteligencia Artificial

Implementacion de arbol de decision
Referencia: https://www.youtube.com/watch?v=sgQAhG5Q7iY
'''

import pandas as pd
import numpy as np

class Node:
    def __init__(
        self,
        feature_index=None,
        tresshld=None,
        left=None,
        right=None,
        info_gain=None,
        value=None
    ) -> None:
        self.feature_index:int = feature_index,
        self.tresshld:int = tresshld,
        self.left:self = left,
        self.right:self = right,

        self.info_gain = info_gain,
        self.value = value

class DesicionTree:
    def __init__(self, max_depth:int=2, min_split:int=2) -> None:
        self.root = None

        self.min_split:int = min_split
        self.max_depth:int = max_depth

    def _fill_tree(self, dataset, actual_depth=0) -> Node:
        ''' El dataframe debe tener la variable objetivo en la ultima columna '''

        X = dataset[:, :-1]
        y = dataset[:, -1]
        sample_count, feature_count = np.shape(X)

        # Hacer split al dataset hasta que se cumplan las condiciones
        if sample_count >= self.min_split and actual_depth <= self.max_depth:
            split = self._getSplit(dataset, feature_count)

            if split['info_gain'] > 0:
                left_child = self._fill_tree(split['data_left'], actual_depth + 1)
                right_child = self._fill_tree(split['data_right'], actual_depth + 1)
                return Node(split['feature_index'], split['threshold'], left_child, right_child)

        # Si el nodo actual es una hoja
        leaf = self._getLeaf(y)
        return Node(value=leaf)

    def _getSplit(self, dataset, feature_count:int) -> dict:
        split:dict = {}
        max_gain = -9999999999

        for index in range(feature_count):
            values = dataset[:, index]
            thresholds = np.unique(values)

            for threshold in thresholds:
                data_left, data_right = self._split(dataset, index, threshold)

                if len(data_left) > 0 and len(data_right) > 0:
                    y = dataset[:, -1]
                    y_left = data_left[:, -1]
                    y_right = data_right[:, -1]

                    info_gain = self._info_gain(y, y_left, y_right)

                    if info_gain > max_gain:
                        split['feature_index'] = index
                        split['threshold'] = threshold
                        split['data_left'] = data_left
                        split['data_right'] = data_right
                        split['info_gain'] = info_gain
                        max_gain = info_gain
        
        return split

    def _split(self, dataset, feature_index, threshold):
        data_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        data_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return data_left, data_right
    
    def _info_gain(self, parent, left, right):
        left_w = len(left) / len(parent)
        right_w = len(right) / len(parent)

        gain = self._gini_index(parent) - (left_w * self._gini_index(left) + right_w * self._gini_index(right))
        return gain

    def _gini_index(self, y):
        labels = np.unique(y)
        gini = 0

        for clas in labels:
            pi = len(y[y == clas] / len(y))
            gini += pi**2
        return 1 - gini
    
    def _gini_index(self, y):
        labels = np.unique(y)
        entropy = 0

        for clas in labels:
            pcls = len(y[y == clas]) / len(y)
            entropy += -pcls * np.log2(pcls)
        return entropy
    
    def _getLeaf(self, y):
        y = list(y)
        return max(y, key=y.count)
    
    def fit(self, X, y):
        dataset = np.concatenate((X, y), axis=1)
        self.root = self._fill_tree(dataset)

    def predict(self, X):
        return [ self.make_prediction(x, self.root) for x in X ]

    def make_prediction(self, x, tree:Node):
        tree = tree[0] if type(tree) == tuple else tree
        if tree.value is not None: return tree.value 

        value = x[tree.feature_index]
        if value <= tree.tresshld:
            return self.make_prediction(x, tree.left)
        
        return self.make_prediction(x, tree.right)
    
    
