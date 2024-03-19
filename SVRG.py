# ライブラリのインポート
import matplotlib.pyplot as plt
import numpy as np
import random
from copy import copy
import SGD_tools as tls
from SGD import SGD

def SVRG(w_0, X, Y, w_opt, n, d, m, s, stepsize, lam, his, dis0): #出力を最後の一項で出すバージョン
    # w_0:\tilde{w}のこと, つまりm回ごとに取られるリファレンスの点
    # X, Y:サンプルの集合でXはd次元n個のデータ, Yはn次元のベクトル
    # w_opt:最適解
    # n, d:サンプル数と次元
    # m:リファレンスを変えるまでの回数で大抵は2n
    # s:反復回数をmで割ったもの
    # stepsize:学習率,ステップサイズ
    # lam:正則化の定数
    # his:今までのdis/dis0を全て記録しておくリスト
    # dis0:初期点と最適解との距離
    if s == 0:
        return his
    else:
        mu = np.array([0]*d, dtype="float")
        for i in range(n):
            mu = mu + tls.loss_grad(w_0, X[i], Y[i]) / n
        w = w_0
        for i in range(m):
            j = np.random.randint(n)
            x = X[j]
            y = Y[j]
            w = w - stepsize * (tls.loss_grad(w, x, y) - tls.loss_grad(w_0, x, y) + mu)
            w = tls.prox2(w, lam)
            dis = np.linalg.norm(w_opt - w)
            his.append(dis/dis0)
            print(i, s, end="\r")
        return SVRG(w, X, Y, w_opt, n, d, m, s-1, stepsize, lam, his, dis0)

def SVRG1(w_m, w, X, Y, w_opt, n, d, m, s, stepsize, lam, his, Rhis, dis0): #出力を平均で出すバージョン
    # w_m:そのときまでの点の平均を出す
    # w:更新するべき点
    # X, Y:サンプルの集合でXはd次元n個のデータ, Yはn次元のベクトル
    # w_opt:最適解
    # n, d:サンプル数と次元
    # m:リファレンスを変えるまでの回数で大抵は2n
    # s:反復回数をmで割ったもの
    # stepsize:学習率,ステップサイズ
    # lam:正則化の定数
    # his:今までのdis/dis0を全て記録しておくリスト
    # dis0:初期点と最適解との距離
    if s == 0:
        return his
    else:
        new_w = np.array([0]*d, dtype="float")
        mu = np.array([0]*d, dtype="float")
        for i in range(n):
            mu = mu + tls.loss_grad(w_m, X[i], Y[i]) / n
        for i in range(m):
            j = np.random.randint(n)
            x = X[j]
            y = Y[j]
            w = w - stepsize * (tls.loss_grad(w, x, y) - tls.loss_grad(w_m, x, y) + mu)
            w = tls.prox2(w, lam*stepsize)
            dis = np.linalg.norm(w_opt - w)
            his.append(dis/dis0)
            "Rhis.append(tls.loss_mean(w, X, Y))"
            new_w = new_w + w / m
            print(i, s, end="\r")
        return SVRG1(new_w, w, X, Y, w_opt, n, d, m, s-1, stepsize, lam, his, Rhis, dis0)



n = 10000
d = 10
data = tls.dataset(n, d)
stepsize = 1 / (2*data.Lip)
lam = 1e-6
N = 20*n # 反復回数

# 初期値をすべて1のベクトルを取る
# これ以降
# 1 -> classicalSGD(SGD.py)
# 2 -> SVRG
# 3 -> SVRGm
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
    w1 = SGD(w1, fj.loss_grad(x, y), stepsize, lam, 0)
    # 距離を求めてdist_hisを更新
    dist = np.linalg.norm(data.opt - w1)
    dist_his1.append(dist/dist0)
    print(i, tls.dig_cut(dist/dist0, 5), end="\r")

m = 2*n # インナーループ一回で行われる更新の回数
w2 = np.array([1]*d, dtype="float")
w3 = np.array([1]*d, dtype="float")
dist_his2 = [1]
dist_his3 = [1]
w_m2 = copy(w2)
mu = np.array([0]*d, dtype="float")
f0 = tls.func(w_m2)
for i in range(n):
    mu += f0.loss_grad(data.X[i], dataY[i])
for i in range(N):
    # 勾配に用いるサンプルを取る
    j = np.random.randint(n)
    x = data.X[j]
    y = data.Y[j]
    fj = tls.func(w1)
    # wを更新
    w1 = SGD(w1, fj.loss_grad(x, y), stepsize, lam, 0)
    # 距離を求めてdist_hisを更新
    dist = np.linalg.norm(data.opt - w1)
    dist_his1.append(dist/dist0)
    print(i, tls.dig_cut(dist/dist0, 5), end="\r")



"""
# メモリを作る, アウターループ5回ごとにメモリを作成
zerotos = []
for i in range((s//5)+1):
    zerotos.append(i)
pre_memori = []
new_memori = []
for z in zerotos:
    pre_memori.append(5*z*m)
    new_memori.append(str(5*z))

# グラフを描画

# 使う方だけコメントアウト消す
plt.title("n="+str(n)+", d="+str(d))
plt.xticks(pre_memori, new_memori)

plt.xlabel(r"$t/m$")
plt.ylabel(r"$\||\theta^{(t)} - \theta^{\ast} \|| / \||\theta^{(0)} - \theta^{\ast} \||$")
plt.yscale('log')

plt.plot(dis1, label="Classical SGD")
plt.plot(dis2, label="SVRG")

plt.plot(dis3, label="SVRG型提案手法" + r"$(\alpha=0.1)$")
plt.plot(dis4, label="SVRG型提案手法" + r"$(\alpha=0.5)$")
plt.plot(dis5, label="SVRG型提案手法" + r"$(\alpha=0.9)$")
plt.legend(prop={"size": 10, "family":"MS Gothic"})
plt.show()
"""