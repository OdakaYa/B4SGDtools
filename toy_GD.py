# f(x, y)=1/2x^2 + 7/3y^2 + 2xy -5x -12y +16の最小値を勾配法によって求める.
# 平方完成すると最小解(1, 3)
# ライブラリのインポート
import matplotlib.pyplot as plt
import numpy as np

#グリッドと初期値の設定
x_min = -25
x_max = 25
y_min = -25
y_max = 25
xs = np.linspace(x_min, x_max, 1000)
ys = np.linspace(y_min, y_max, 1000)
X, Y = np.meshgrid(xs, ys)
#関数の設定
Z = 1/2*X**2 + 7/3*Y**2 + 2*X*Y - 5*X - 12*Y + 16
opt_x = -1
opt_y = 3

stepsize = 0.35
x_init = 20
y_init = -20
x_list = [x_init]
y_list = [y_init]

# 入力した座標と最適解との距離を計算
def get_dis(x, y):
    return (x - opt_x)**2 + (y - opt_y)**2

# 最急降下法を用いたときの(x, y)の次の座標
def get_grad(x, y):
    return [x + 2*y - 5, 14/3*y + 2*x - 12]

i = 0
x = x_init
y = y_init
dis = get_dis(x, y)
while dis > 0.01:
    grad = get_grad(x, y)
    x = x - stepsize * grad[0]
    y = y - stepsize * grad[1]
    x_list.append(x)
    y_list.append(y)
    dis = get_dis(x, y)
    print(i, int(x*1000)/1000, int(y*1000)/1000, dis, end="\r")
    i += 1

# 各反復の座標をプロット
plt.plot(x_list, y_list, color= "red")
# グラフを描画
plt.contour(X, Y ,Z ,levels=20 , colors=['black'])
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('x')
plt.ylabel('y')
plt.show()