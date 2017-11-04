from functools import partial
import argparse
import sympy as sp
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from symdoc import Markdown, doit
#from sympy.utilities.lambdify import lambdastr
cmdline_parser = argparse.ArgumentParser()
Markdown.add_parser_option(cmdline_parser)


M = Markdown('kk', title='Kamada Kawai Algorithm')
markdown, cmdline = M.markdown, M.cmdline

# gradient と Hessian
dim = 2

n = sp.Symbol('n' , integer = True)
# ある頂点iについて考えるとする
# よってここのIndexedBaseはlambdifyした後にiとの差を受け取ることになる

Pi = sp.IndexedBase('P')
Li = sp.IndexedBase('L')
Ki = sp.IndexedBase('K')

# これはiベクトル
p = sp.IndexedBase('p')

j, d = [sp.Idx(*spec) for spec in [('j',n-1), ('d',dim)]]
j_range, d_range = [(idx, idx.lower, idx.upper) for idx in [j, d]]
length_j = sp.sqrt(sp.Sum((Pi[j,d]-p[d])**2, d_range)).doit()
potential_j = Ki[j] * (length_j - Li[j]) ** 2 /2
function_E = sp.Sum(potential_j,j_range)
GRAD_0 = sp.simplify(sp.diff(function_E,p[0]))
GRAD_1 = sp.simplify(sp.diff(function_E,p[1]))
HESS_00 = sp.simplify(sp.diff(GRAD_0,p[0]))
HESS_01 = sp.simplify(sp.diff(GRAD_0,p[1]))
HESS_10 = sp.simplify(sp.diff(GRAD_1,p[0]))
HESS_11 = sp.simplify(sp.diff(GRAD_1,p[1]))
grad_0,grad_1 = [sp.lambdify((Pi, Li, Ki, p, n),f, dummify = False) for f in [GRAD_0, GRAD_1]]
hess_00, hess_01, hess_11 = [sp.lambdify((Pi, Li, Ki, p, n),f, dummify = False) for f in [HESS_00, HESS_01, HESS_11]]

# ループ内部の数値計算、線形方程式を解く
def cal_loop(Pd, Ld, Kd, pd, nd):
	a_grad = ([-grad_0(Pd, Ld, Kd, pd, nd),-grad_1(Pd, Ld, Kd, pd, nd)])
	a_hess = ([[hess_00(Pd, Ld, Kd, pd, nd), hess_01(Pd, Ld, Kd, pd, nd)],[hess_01(Pd, Ld, Kd, pd, nd),hess_11(Pd, Ld, Kd, pd, nd)]])
	return np.linalg.solve(a_hess,a_grad)

def slice_LK(Matrix, n):
	r_Matrix = np.zeros([n,n-1])
	for i in range(n):
		Matrix_d1 = Matrix[i][:i]
		Matrix_d2 = Matrix[i][i+1:]
		Matrix_d = np.concatenate((Matrix_d1,Matrix_d2), axis = 0)
		r_Matrix[i] = Matrix_d
	return r_Matrix

def slice_P(P, i):
	Pd1 = P[:i]
	Pd2 = P[i+1:]
	Pd = np.concatenate((Pd1,Pd2), axis = 0)
	return (Pd)

def check_max(P, L, K, n):
	delta_max = 0
	Maxi = -1
	for i in range(n):
		Pd = slice_P(P,i)
		delta = np.sqrt(grad_0(Pd, L[i], K[i], P[i], n)**2 + grad_1(Pd, L[i], K[i], P[i], n)**2)
		if (delta_max < delta):
			delta_max = delta
			Maxi = i
	return [Maxi, delta]

def makeMaterix(vlis, conslis):
	n = len(vlis)
	length_cons = len(conslis)
	cons_matrix = np.zeros([n,n])
	for i in range(length_cons):
		for j in conslis[i]:
			for k in conslis[i]:
				if not (j == k):
					cons_matrix[j][k] = 1
	return cons_matrix

def warshallFloyd(cons_matrix):
    n = cons_matrix.shape[1]
    distance_matrix = -n*cons_matrix + n+1
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if(distance_matrix[i][j] > distance_matrix[i][k] + distance_matrix[k][j] ):
                    distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]

    return distance_matrix

def draw_graph(cons_matrix,P,n,name):
	figname = 'pics/' + name + '.png'
	for i in range(n-1):
		for j in range(i,n):
			if(cons_matrix[i,j] == 1):
				plt.plot([P[i,0],P[j,0]],[P[i,1],P[j,1]],'k-')
	plt.plot(P.T[0],P.T[1],'go')
	plt.title(name)
	plt.axes().set_aspect('equal', 'datalim')
	plt.savefig(figname)
	plt.clf()
	#plt.show()

def kk_algo(vlis, conslis, name, L0=10, K0=10, eps=0.0001):
	n = len(vlis)
	cons_matrix = makeMaterix(vlis,conslis)
	D = warshallFloyd(cons_matrix)
	P = np.zeros([n,dim])
	for i in range(n):
		P[i,0] = L0 * np.cos(2*np.pi/n*i)
		P[i,1] = L0 * np.sin(2*np.pi/n*i)
	Kf = K0 * D**(-2) * (-np.eye(n)+np.ones([n,n]))
	D = D * (-np.eye(n) + np.ones([n,n]))
	Lf = L0 * D
	L = slice_LK(Lf, n)
	K = slice_LK(Kf, n)
	print(P)
	#print(K)
	#print(L)
	draw_graph(cons_matrix,P,n, name+"_default")
	delta_max = 10000.0
	while (delta_max > eps):
		Maxi, delta_max = [max for max in check_max(P, L, K, n)]
		delata_p = cal_loop(slice_P(P,Maxi), L[Maxi], K[Maxi], P[Maxi], n)
		P[Maxi] = P[Maxi] + delata_p
	draw_graph(cons_matrix,P,n, name+"_moved")
	return P

def test_KK(lis,conslis,name):
	P = kk_algo(lis,conslis,name)
	print(P)
	markdown(
r'''
![最適化前](pics/{name}_default.png)
![最適化後](pics/{name}_moved.png)
''', **locals())

@doit
def __document__():
	n = sp.Symbol('n',integer = True)
	# ある頂点iについて考えるとする
	# よってここのIndexedBaseはlambdifyした後にiとの差を受け取ることになる

	Pi = sp.IndexedBase('P')
	Li = sp.IndexedBase('L')
	Ki = sp.IndexedBase('K')

	# これはiベクトル
	p = sp.IndexedBase('p')

	j, d = [sp.Idx(*spec) for spec in [('j',n-1), ('d',dim)]]
	j_range, d_range = [(idx, idx.lower, idx.upper) for idx in [j, d]]
	length_j = sp.sqrt(sp.Sum((Pi[j,d]-p[d])**2, d_range)).doit()
	potential_j = Ki[j] * (length_j - Li[j]) ** 2 /2
	function_E = sp.Sum(potential_j,j_range)
	E = sp.Function('E')
	dEp0 = sp.diff(E(p[0],p[1]), p[0])
	dEp1 = sp.diff(E(p[0],p[1]), p[1])
	dEp00 = sp.diff(dEp0, p[0])
	dEp01 = sp.diff(dEp0, p[1])
	dEp10 = sp.diff(dEp1, p[0])
	dEp11 = sp.diff(dEp1, p[1])
	delta_i = sp.sqrt(dEp0**2 + dEp1**2)
	markdown(
r'''
# KK法による無向グラフの描画

無向グラフの描き方として知られているKamada-Kawai法のアルゴリズムを実装する。

全ての頂点がバネによって繋がっているものとしてグラフを描く。バネのエネルギーが
安定している状態のグラフを描いていく。
グラフ全体を一つの系としてみなして、その系のエネルギーが低くなるように、頂点を動かしていく。
まず、ある頂点$v_i$の位置ベクトルを$p$とする。この時、ある頂点$v_j$との距離は、
$${{length}}_j={length_j}$$
で定義される。$v_i$と$v_j$をつなぐバネのポテンシャルはその自然長を$L_j$、バネ係数を$K_j$とすると
$${{potential_j}}={potential_j}$$
と定義できる。よって、$v_i$においてのポテンシャル$E$は、
$${{E}}={function_E}$$
バネの自然長は、頂点$v_i$と$v_j$のグラフ上の距離を$d_{{i,j}}$として
$$L_j=L_0 \times d_{{i,j}}$$と定める。また、バネ係数はそれぞれ
$$K_j=K_0 / d_{{i,j}}^2$$と定める。$K_0$と$L_0$は定数である。
次に、このエネルギーを最小化していき、安定な状態を求める。
そのために、$p=P_i$と定め、$p$だけを変数とみて、考える。
この$p$は各$v_i$の中で
$$\Delta_i = {delta_i}$$
が最大の$i$とする。そして、$p$の新しい座標を
$$p_0 = p_0 + \delta p_0$$
$$p_1 = p_1 + \delta p_1$$
とする。この$\delta p_0$、$\delta p_1$は次の線形方程式の解である
$${dEp00} \delta p_0 + {dEp01} \delta p_1 = -{dEp0}$$
$${dEp10} \delta p_0 + {dEp11} \delta p_1 = -{dEp1}$$
これを繰り返し、$\Delta_i$の最大値が十分小さくなるまで繰り返す。
以下に、このアルゴリズムを適用する前の図と、適用後の図を示す。



''',**locals())
lis1 = [0,1,2,3]
conslis1 = [[0,1,2],[2,3]]
lis2 = [0,1,2,3,4,5]
conslis2 = [[0,1],[0,2],[0,3],[0,4],[1,2],[1,4],[1,5],[2,3],[2,5],[3,4],[3,5],[4,5]]
lis3 = [0,1,2,3,4,5,6,7,8,9,10,11]
conslis3 = [[0,1],[1,5],[2,3,4,5],[5,7],[6,7],[7,8],[6,8],[8,9],[9,10],[10,11],[8,11]]
lis4 =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
conslis4 = [[0,1,2,3],[3,4],[4,5,6,7],[4,8],[8,9,10,11],[8,12],[12,13,14,15],[3,12]]
lis5 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
conslis5 = [[0,1],[0,4],[0,5],[1,2],[1,7],[2,3],[2,9],[3,4],[3,11],[4,13],[5,6],[5,14],[6,7],[6,15],[7,8],[8,9],[8,16],[9,10],[10,11],[10,17],[11,12],[12,13],[12,18],[13,14],[14,19],[15,16],[15,19],[16,17],[17,18],[18,19]]

test_KK(lis1,conslis1,"test1")
test_KK(lis2,conslis2,"test2")
test_KK(lis3,conslis3,"test3")
test_KK(lis4,conslis4,"test4")
test_KK(lis5,conslis5,"test5")


'''
	GRAD_0_s = "\frac{{\partial E}}{{\partial p_0}}"
	GRAD_1_s = "\frac{{\partial E}}{{\partial p_1}}"
	HESS_00_s = "\frac{{\partial^2 E}}{{\partial^2 p_0}}"
	HESS_00_s = "\frac{{\partial^2 E}}{{\partial p_0 \partial p_1}}"
	HESS_00_s = "\frac{{\partial^2 E}}{{\partial^2 p_1}}"
def type_check(sym_object):
	print(type(sym_object))
	

@doit
def __test__():
	length_j = sp.sqrt(sp.Sum((Pi[j,d]-p[d])**2, d_range)).doit()
	potential_j = Ki[j] * (length_j - Li[j]) ** 2 /2
	function_E = sp.Sum(potential_j,j_range)
	GRAD_0 = sp.diff(function_E,p[0])
	GRAD_1 = sp.diff(function_E,p[1])
	HESS_00 = sp.diff(GRAD_0,p[0])
	HESS_01 = sp.diff(GRAD_0,p[1])
	HESS_10 = sp.diff(GRAD_1,p[0])
	HESS_11 = sp.diff(GRAD_1,p[1])
	grad_0,grad_1 = [sp.lambdify((Pi, Li, Ki, p, n),f, dummify = False) for f in [GRAD_0, GRAD_1]]
	hess_00, hess_01, hess_11 = [sp.lambdify((Pi, Li, Ki, p, n),f, dummify = False) for f in [HESS_00, HESS_01, HESS_11]]

	L0=10
	K0=10
	ni = 4
	A = makeMaterix([0,1,2,3],[[0,1,2],[2,3]])
	D = warshallFloyd(A)
	P = np.zeros([ni ,dim])
	for i in range(ni):
		P[i,0] = L0 * np.cos(2*np.pi/ni*i)
		P[i,1] = L0 * np.sin(2*np.pi/ni*i)
	Kf = K0 * D**(-2) * (-np.eye(ni)+np.ones([ni,ni]))
	D = D * (-np.eye(ni) + np.ones([ni,ni]))
	Lf = L0 * D
	type_check(length_j)
	type_check(potential_j)
	type_check(function_E)
	type_check(GRAD_0)
	type_check(HESS_00)
	type_check(grad_0)
	type_check(Pi)
	type_check(Li)
	type_check(n)
	type_check(Ki)
	type_check(p)

	L = slice_LK(Lf, ni)
	K = slice_LK(Kf, ni)
	Pd = slice_P(P, 1)

	L1 = L[1]
	K1 = K[1]
	P1 = P[1]
	a = lambdastr((Pi, Li, Ki, p, n),GRAD_0)
	b = grad_0(Pd, L1, K1, P1, 4)
	markdown(
r
$${length_j}$$
$${potential_j}$$
$${function_E}$$
$${GRAD_0}$$
$${GRAD_1}$$
$${HESS_00}$$
$${HESS_01}$$
$${HESS_10}$$
$${HESS_11}$$
$${A}$$
$${D}$$
$${a}$$
$${L}$$
$${K}$$
$${L1}$$
$${K1}$$
$${P1[0]}$$
$${P1[1]}$$
$${Pd[0][0]}$$
$${Pd[0][1]}$$
$${Pd[1][0]}$$
$${Pd[1][1]}$$
$${Pd[2][0]}$$
$${Pd[2][1]}$$
'''
