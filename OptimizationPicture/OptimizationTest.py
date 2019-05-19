# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def display(source1, source2):
    with open(source1, "rb") as fsource1, open(source2,"rb") as fsource2:
        X = np.load(fsource1)
        y = np.load(fsource2)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=20)
        print(X)
        print(y)
        plt.show()


def expand(x):
    """
    :param x: 特征文件路径。
    :return: 经过扩张的特征矩阵。
    """
    with open(x, "rb") as fsource1:
        X = np.load(fsource1)
        X_expanded = np.zeros((X.shape[0], 6))
        X_expanded[:, 0], X_expanded[:, 1] = X[:, 0], X[:, 1]
        X_expanded[:, 2], X_expanded[:, 3] = X[:, 0]**2, X[:, 1]**2
        X_expanded[:, 4], X_expanded[:, 5] = X[:, 0] * X[:, 1], np.ones(X.shape[0])
        return X_expanded


def probability(X, W):
    """
    :param X: feature matrix X of shape [n_sample,6]
    :param W: weight vector w of shape [6] for each of the expanded features
    :return: 返回给定x,其中预测结果是1所对应的概率。
    """
    z = np.dot(X, W)
    a = 1./(1+np.exp(-z))
    return a


def compute_score(features_matrix, weight):
    out = probability(features_matrix, weight)
    return out


if __name__ == "__main__":
    X_expand = expand("train.npy")
    W = np.linspace(-1, 1, 6)
    out = compute_score(X_expand, W)
    print(out)
    print(len(out))

