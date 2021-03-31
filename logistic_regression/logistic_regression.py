import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt 
from sklearn import linear_model

def sigmoid(x):
    return 1/(1+np.exp(-x))

def loss_func(X, y, beta):
    '''
    最大似然损失：L(beta) = sigma(from i to N) [- y_i * (beta*x_i) + log(1 + exp(beta*x_i))];
    '''
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    beta = beta.reshape(-1, 1)
    y = y.reshape(-1, 1)

    linear_res = np.dot(X_hat, beta)
    Loss = -y*linear_res + np.log(1+np.exp(linear_res))

    return Loss.sum()

def gradient(X, y, beta):
    '''
    使用最大似然损失，对beta求梯度
    '''
    X_hat = np.c_[X,np.ones((X.shape[0], 1))] # 17 * 3
    beta = beta.reshape(-1, 1)                # 3 * 1
    y = y.reshape(-1 ,1)                      # 17 * 1
    p1 = sigmoid(np.dot(X_hat, beta))         # 17 * 1

    gra = (-X_hat * (y - p1)).sum(0)          # 1 * 3

    return gra.reshape(-1 , 1)                # 3 * 1

def initialize_beta(n):
    beta = np.random.randn(n+1, 1)*0.5 + 1
    return beta

def update_parameters_gradDesc(X, y, beta, learning_rate, num_iterations, print_cost):
    '''
    采用梯度下降法更新参数，损失函数使用最大似然损失
    '''
    for i in range(num_iterations):
        grad = gradient(X, y, beta)
        beta = beta - learning_rate*grad

        if(i % 10 == 0) & print_cost:
            print('{}th iteration, cost is {}'.format(i, loss_func(X, y, beta)))
    
    return beta

def logistic_model(X, y, num_iterations=100, learning_rate=1.2, print_cost=False, method='gradDesc'):
    '''
    :param X: 训练数据
    :param y: 标注结果
    '''

    m, n = X.shape
    beta = initialize_beta(n)

    if method == 'gradDesc':
        return update_parameters_gradDesc(X, y ,beta, learning_rate, num_iterations, print_cost)
    else:
        raise ValueError("Unkonwn Solver %s" % method)

def predict(X, beta):
    X_hat = np.c_[X, np.ones((X.shapep[0], 1))]
    p1 = sigmoid(np.dot(X_hat, beta))

    p1[p1 >= 0.5] = 1
    p1[p1 < 0.5] = 0

    return p1

if __name__ == '__main__':
    data_path = "watermelon3_0_Ch.csv"
    data = pd.read_csv(data_path).values

    is_good = data[:, 9] == '是'
    is_bad = data[:, 9] == '否'

    X = data[:, 7:9].astype(float)
    y = data[:, 9]

    y[y == '是'] = 1
    y[y == '否'] = 0
    y = y.astype(int)
    plt.scatter(data[:,7][is_good], data[:, 8][is_good], c ='k', marker='o')
    plt.scatter(data[:,7][is_bad], data[:, 8][is_bad], c = 'r', marker = 'x')

    plt.xlabel('密度')
    plt.ylabel('含糖量')

    beta = logistic_model(X, y, print_cost = True, method='gradDesc', learning_rate=0.3, num_iterations=1000)
    w1, w2, intercept = beta
    x1 = np.linspace(0,1)
    y1 = -(w1*x1 + intercept)/w2

    ax1, = plt.plot(x1, y1, label = r'my_logistic_gradDesc')

    lr = linear_model.LogisticRegression(solver = 'lbfgs', C = 1000)
    lr.fit(X, y)

    lr_beta = np.c_[lr.coef_, lr.intercept_]
    print(loss_func(X,y, lr_beta))

    w1_sk, w2_sk = lr.coef_[0, :]

    x2 = np.linspace(0,1)
    y2 = -(w1_sk*x2 + lr.intercept_)/w2

    ax2, = plt.plot(x2, y2, label = 'sklearn_logistic')

    plt.legend(loc='upper right')
    plt.show()
