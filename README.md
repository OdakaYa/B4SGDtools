# B4SGDtools

これは学部四年時に使っていた研究用コードの保管場所になります。
当時GitHubでコードを保管する習慣がなかったため現在(2024/3/14)更新途中となります。

SGD_tools.py
確率的勾配法の実験に使っていた関数やclassをまとめたものになります。これらをライブラリとして活用することで実験を行っていました。

toy_poly.py
確率的勾配法を用いて一変数関数(多項式関数)のパラメータを予測するプログラムになります。

toy_GD.py
確率的勾配法の源流となる最急降下法の様子がよくわかる実験になります。グラフ上の点が最適点に近づいていくことがわかります。

SGD.py
最も古典的な確率的勾配法(SGD)を行う実験になります。最急降下法で使われる傾きの項をランダムに取った一つの点での傾きで代用することで計算量を削減する方法です。

SVRG.py
こちらも確率的勾配法の一種でSGD.pyよりも理論上分散が少なくなる形式のものになります。
