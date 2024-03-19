# ライブラリのインポート
import matplotlib.pyplot as plt
import numpy as np
import random
from copy import copy
import SGD_tools as tls
from SGD import SGD

class SVRG(SGD):
    def __init__(self, s, l, m, w):
        super().__init__(s, l)
        self.m = m
        self.f0 = tls.func(w)
    
    def mu(self, X, Y):
        self.mu = np.array([0]*d, dtype="float")
        N = len(X)
        f0 = self.f0
        for i in range(N):
            self.mu += f0.loss_grad(X[i], Y[i]) / N
    
    def new_w(self, w, grad, ch):
        return super().new_w(w, grad - self.f0.loss_grad(x, y) + self.mu, ch)
    
    def update_wm(self, w, n):
        m = self.m
        if n % m == 0:
            self.f0 = tls.func(w)

n = 10000
d = 10
data = tls.dataset(n, d)
stepsize = 1 / (2*data.Lip)
lam = 1e-6
N = 20*n # 反復回数

# これ以降
# 1 -> classicalSGD(SGD.py)
# 2 -> SVRG
# 3 -> SVRGm

# 初期値をすべて1のベクトルを取る
sgd = SGD(stepsize, lam)
w1 = np.array([1]*d, dtype="float")
# 最適点と初期値との距離を計算する.あとで使う.
dist0 = np.linalg.norm(data.opt - w1)
# dis0と現在の最適点との距離をリストにする.
dist_his1 = [1]
for i in range(N):
    # 勾配に用いるサンプルを取る
    j = np.random.randint(n)
    x = data.X[j]
    y = data.Y[j]
    fj = tls.func(w1)
    # wを更新
    w1 = sgd.new_w(w1, fj.loss_grad(x, y), 2)
    # 距離を求めてdist_hisを更新
    dist = np.linalg.norm(data.opt - w1)
    dist_his1.append(dist/dist0)
    print(i, tls.dig_cut(dist/dist0, 5), end="\r")

m = 2*n # インナーループ一回で行われる更新の回数
w2 = np.array([1]*d, dtype="float")
w3 = np.array([1]*d, dtype="float")
dist_his2 = [1]
dist_his3 = [1]
svrg = SVRG(stepsize, lam, m, w2)
svrg.mu(data.X, data.Y)
for i in range(N):
    # 勾配に用いるサンプルを取る
    j = np.random.randint(n)
    x = data.X[j]
    y = data.Y[j]
    fj = tls.func(w2)
    # wを更新
    w2 = svrg.new_w(w2, fj.loss_grad(x, y), 2)
    # 距離を求めてdist_hisを更新
    dist = np.linalg.norm(data.opt - w2)
    dist_his2.append(dist/dist0)
    print(i, tls.dig_cut(dist/dist0, 5), end="\r")
    if i+1 % m == 0:
        svrg.f0 = tls.func(w2)
        svrg.mu(data.X, data.Y)

# グラフを描画
plt.yscale('log')
plt.minorticks_on()
plt.plot(dist_his1)
plt.plot(dist_his2)
plt.show()