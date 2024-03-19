# ライブラリのインポート
import numpy as np
import random
from scipy.sparse.linalg import eigs

#一般の線形回帰モデルを考える
class func:
    def __init__(self, A:list[float]):
        #A：係数リスト（index0の値は定数項）
        #dig：入力データxの次元の数
        self.coef = np.array(A)
        self.dig = len(A)
    
    def get(self, x:list[float]) -> float: #f(x)の値を取得する
        return np.dot(self.coef, x)

    def SE(self, x:list[float], y:float) -> float: #二乗誤差を計算
        return (self.get(x) - y)**2

    def MSE(self, x:list[list[float]], y:list[float]) -> float: #平均二乗誤差を計算
        res = 0
        n = len(y)
        for i in range(n):
            res += self.SE(x[i], y[i])
        return res / n

    def loss_grad(self, x:list[float], y:float) -> list[float]: #損失関数の勾配を計算
        n = len(x)
        res = [None] * n
        temp = self.get(x) - y
        for i in range(n):
            res[i] = 2 * x[i] * temp
        return np.array(res)

#係数リストAに対してf(x) = A[0] + A[1]x +A[2]x^2 + A[3]x^3 + ...の場合（多項式関数）
class poly_func(func):
    def __init__(self, A:list[float]):
        super().__init__(A)

    def get_poly(self, x:float) -> float:
        n = self.dig
        temp = [None]*n
        for i in range(n):
            temp[i] = x**i
        return super().get(temp)

    def SE_poly(self, x:float, y:float) -> float:
        return (self.get_poly(x) - y)**2

    def MSE_poly(self, x:list[float], y:list[float]) -> float:
        res = self.coef[0]
        n = len(y)
        for i in range(n):
            res += self.SE_poly(x[i], y[i])
        return res / n

    def loss_grad_poly(self, x:float, y:float) -> list[float]: #損失関数の勾配を計算
        n = self.dig
        temp = [None]*n
        for i in range(n):
            temp[i] = x**i
        return super().loss_grad(temp, y)

#実験に使うデータセットを生成
class dataset:
    def __init__(self, n:int, d:int):
        self.opt = np.random.randn(d) #幅を持った値で試すと良し
        self.X = np.random.normal(0, 5, (n, d))
        temp = []
        temp_f = func(self.opt)
        for x in self.X:
            temp.append(temp_f.get(x) + np.random.randn())
        self.Y = np.array(temp)
        A = np.matmul((self.X).T, self.X)
        self.Lip = 2 * eigs(A, 1)[0].real

#L1正則化(定数g)
def prox1(w:list[float], g:float) -> list[float]:
    n = len(w)
    new_w = [None]*n
    for i in range(n):
        temp = w[i]
        if temp > g:
            new_w[i] = temp - g
        elif temp < -g:
            new_w[i] = temp + g
    return new_w

#L2正則化(定数g)
def prox2(w:list[float], g:float) -> list[float]:
    return w * (1/(1+g))

###出力を見やすくする関数(d桁まで表示)
def dig_cut(val, d):
    return int(val*10**d)/10**d