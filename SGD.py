# ライブラリのインポート
import matplotlib.pyplot as plt
import numpy as np
import random
import SGD_tools as tls

class SGD():
    def __init__(self, s, l):
        self.stepsize = s
        self.lam = l

    def new_w(self, w, grad, ch):
        w = w - stepsize * grad
        if ch == 1:
            w = tls.prox1(w, lam)
        elif ch == 2:
            w = tls.prox2(w, lam)
        return w

if __name__ == "__main__":
    n = 10000
    d = 50
    data = tls.dataset(n, d)
    stepsize = 1 / (2*data.Lip)
    lam = 1e-6
    sgd = SGD(stepsize, lam)

    # 初期値をすべて1のベクトルを取る
    w = np.array([1]*d, dtype="float")
    # 最適点と初期値との距離を計算する.あとで使う.
    dist0 = np.linalg.norm(data.opt - w)
    # dis0と現在の最適点との距離をリストにする.
    dist_his = [1]
    for i in range(20*n):
        # 勾配に用いるサンプルを取る
        j = np.random.randint(n)
        x = data.X[j]
        y = data.Y[j]
        fj = tls.func(w)
        # wを更新
        w = sgd.new_w(w, fj.loss_grad(x, y), 2)
        # 距離を求めてdist_hisを更新
        dist = np.linalg.norm(data.opt - w)
        dist_his.append(dist/dist0)
        print(i, tls.dig_cut(dist/dist0, 8), end="\r")

    # グラフを描画
    plt.yscale('log')
    plt.minorticks_on()
    plt.plot(dist_his)
    plt.show()