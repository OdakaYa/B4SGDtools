# ライブラリのインポート
import numpy as np
import random
from scipy.sparse.linalg import eigs

def func(f, x):
    return np.dot(f, x)

def loss(a, x, y):
    return (func(a, x) - y)**2

def loss_mean(a, X, Y):
    val = 0
    n = len(X)
    for i in range(n):
        val += loss(a, X[i], Y[i]) / n
    return val

def loss_grad(a, x, y):
    return 2 * x * (func(a, x) - y)

def loss_var(w, X, Y):
    n = len(Y)
    var = 0
    mean = 0
    for i in range(n):
        mean += loss(w, X[i], Y[i]) / n
    for i in range(n):
        var += (loss(w, X[i], Y[i]) - mean)**2 / n
    return var

def prox(a, gam):
    n = len(a)
    new_a = [0]*n
    for i in range(n):
        if a[i] > gam:
            new_a[i] = a[i] - gam
        elif a[i] < -gam:
            new_a[i] = a[i] + gam
    return new_a

def prox2(a, gam):
    return np.array(a) * (1/(1+gam))

def dataset_aaa(n, d): # 多分もう使わない
    X = np.random.randn(n, d)
    Y = np.array([1]*n, dtype = "float")
    A = np.matmul(X.T, X)
    B = np.matmul(X.T, Y)
    W = np.matmul(np.linalg.pinv(A), B)
    return X, Y, W

def dataset(n, d):
    w = np.random.randn(d) #幅を持った値で試すと良し
    X = np.random.normal(0, 5, (n, d))
    Y = []
    for x in X:
        Y.append(func(w, x) + np.random.randn())
    return X, Y, w

def Lip_const(X):
    A = np.matmul(X.T, X)
    return 2 * eigs(A, 1)[0].real

def dig_cut(val, dig):
    return int(val*10**dig)/10**dig