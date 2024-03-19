# ライブラリのインポート
import matplotlib.pyplot as plt
import numpy as np
import SGD_tools as tls

#ステップサイズ
ss = 0.000002

# y = x^3 + 2x^2 - 11x - 12上に点を取る
# -10~10で学習用データを生成
xs = []
ys = []
for i in range(1000):
    r = 20*np.random.rand() - 10
    xs.append(r)
xs = sorted(xs)
a = [-12, -11, 2, 1]
f = tls.poly_func(a)
for x in xs:
    # 標準正規分布に基づくノイズを足す
    ys.append(f.get_poly(x) + np.random.randn())

# 初期値
w = np.array([10, 10, 10, 10], dtype="float")
for i in range(1000000):
    # 勾配に用いるサンプルをひとつ取る
    n = np.random.randint(1000)
    fw = tls.poly_func(w)
    w = w - fw.loss_grad_poly(xs[n], ys[n]) * ss
    print(i, int(w[0]*1000)/1000, int(w[1]*1000)/1000, int(w[2]*1000)/1000, int(w[3]*1000)/1000, end = "\r")

yss = []
fw = tls.poly_func(w)
for x in xs:
    yss.append(fw.get_poly(x))
plt.plot(xs, yss, color= "red")

# グラフを描画
# 青い点がサンプル、赤い線が今回の推定で得られた曲線
plt.scatter(xs, ys)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()