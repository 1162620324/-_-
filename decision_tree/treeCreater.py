import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target
import treePlottter
import pruning

class Node(Object):
    def __init__(self):
        self.feature_name = None
        self.feature_index = None
        self.subtree = {}
        self.impurity = None
        self.is_continuous = False
        self.split_value = None
        self.is_leaf = False
        self.leaf_class = None
        self.leaf_num = None
        self.high = -1

class DecisionTree(object):
    def __init__(self, criterion = 'gini', pruning = None):
        '''
        param criterion: 划分方法选择 
        param pruning: 是否剪枝
        '''
        assert criterion in ('gini', 'infogain','gainratio')
        assert pruning in (None, 'pre_pruning', 'post_pruning')
        self.criterion = criterion
        self.pruning = pruning

    def fit(self, X_train, y_train, X_val = None, y_val = None):
        '''
        生成决策树
        -------------
        param X_train: 只支持DataFrame类型数据，因为DataFrame中已有列名，省去一个列名的参数。
                        不支持np.array等其他数据类型
        param y_train:
        '''
        if self.pruning is not None and (X_val is None or y_val is None):
            raise Exception('Warning!')
        
        X_train.reset_index(inplace=True, drop=True)
        y_train.reset_index(inplace=True, drop=True)

        if X_val is not None:
            X_val.reset_index(inplace=True, drop=True)
            y_val.reset_index(inplace=True, drop=True)
        
        self.cloumns = list(X_train.columns)
        self.tree_ = self.generate_tree(X_train, y_train)

        if self.pruning == 'pre_pruning':
            pruning.pre_pruning(X_train, y_train, X_val, y_val, self.tree_)
        elif self.pruning == 'post_pruning':
            pruning.post_pruning(X_train, y_train, X_val, y_val, self.tree_)
        
        return self

    def generate_tree(self, X, y):
        '''
        辅助生成决策树;
        ---------------
        param X: 训练样本数据
        param y:
        '''

    def predict(self, X):
        '''
        同样只支持pd.DataFrame数据
        param X: 
        '''

    def predict_single(self, x, subtree=None):
        '''
        预测单一样本，写成循环
        param X:
        param subtree: 根据特征，往下递进的子树
        '''

    def choose_best_feature_to_split(self, X, y):
        '''
        选择对应度量方式最佳的划分特征
        '''

    def choose_best_feature_gini(self, X, y):

    def choose_best_feature_infogain(self,X , y):
    
    def choose_best_feature_gainratio(self, X, y):
    
    def gini(self, y):

    def info_gain(self, feature, y, entD, is_continuous=False):

    def info_gainRatio(self, feature, y, entDm, is_continuous=False):

    def entrophy(self, y):

if __name__ == '__main__':

    data_path = r""
    data = pd.read_table(data_path, encoding='utf8', delimiter=',', index_col=0)

    train = [1,2,3,6,7,10,14,15,16,17]
    train = [i-1 for i in train]
    x = data.iloc[train, :6]
    y = data.iloc[train, 6]

    test = [4,5,8,9,11,12,13]
    test = [i-1 for i in test]
    X_val = data.iloc[test, :6]
    y_val = data.iloc[test, 6]

    criterion = 'gini'
    pruning = 'pre_pruning' 
    tree = DecisionTree(criterion, pruning)
    tree.fit(X, y, X_val, y_val)

    print(np.mean(tree.predict(X_val) == y_val)

    treePlotter.create_plot(tree.tree_)


    
    