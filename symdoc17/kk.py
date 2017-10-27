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
n1 = sp.Symbol('n1' , integer = True)
# ある頂点iについて考えるとする
# よってここのIndexedBaseはlambdifyした後にiとの差を受け取ることになる

Pi = sp.IndexedBase('P')
Li = sp.IndexedBase('L')
Ki = sp.IndexedBase('K')

# これはiベクトル
p = sp.IndexedBase('p')

j, d = [sp.Idx(*spec) for spec in [('j',n-1), ('d',dim)]]
j_range, d_range = [(idx, idx.lower, idx.upper) for idx in [j, d]]
lnegth_j = sp.sqrt(sp.Sum((Pi[j,d]-p[d])**2, d_range)).doit()
potential_j = Ki[j] * (lnegth_j - Li[j]) ** 2 /2
function_E = sp.Sum(potential_j,j_range)
GRAD_0 = sp.diff(function_E,p[0])
GRAD_1 = sp.diff(function_E,p[1])
HESS_00 = sp.diff(GRAD_0,p[0])
HESS_01 = sp.diff(GRAD_0,p[1])
HESS_10 = sp.diff(GRAD_1,p[0])
HESS_11 = sp.diff(GRAD_1,p[1])
grad_0,grad_1 = [sp.lambdify((Pi, Li, Ki, p, n),f, dummify = False) for f in [GRAD_0, GRAD_1]]
hess_00, hess_01, hess_11 = [sp.lambdify((Pi, Li, Ki, p, n),f, dummify = False) for f in [HESS_00, HESS_01, HESS_11]]

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
	return Maxi

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

def draw_graph(cons_matrix,P,n):
	Path = mpath.Path
	fig, ax = plt.subplots()
	for i in range(n):
		ax.plot(P[i,0],P[i,1], "ro")
		for j in range(i,n):
			if(cons_matrix[i][j] == 1):
				ax.plot([P[i,0],P[j,0]],[P[i,1],P[j,1]],'k-')
	ax.set_title('The red point should be on the path')
	plt.show()

def kk_algo(vlis, conslis, L0=10, K0=10, eps=0.0000001):
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
	print(K)
	print(L)
	delta_max = 10000.0
	while (delta_max > eps):
		Maxi = check_max(P, L, K, n)
		delta_max = np.sqrt(grad_0(slice_P(P,Maxi), L[Maxi], K[Maxi], P[Maxi], n)**2 + grad_1(slice_P(P, Maxi), L[Maxi], K[Maxi], P[Maxi], n)**2)
		delata_p = cal_loop(slice_P(P,Maxi), L[Maxi], K[Maxi], P[Maxi], n)
		P[Maxi] = P[Maxi] + delata_p

	draw_graph(cons_matrix,P,n)
	return P

@doit
def test_kk1():
	lis = [0,1,2,3]
	conslis = [[0,1,2],[2,3]]

	P = kk_algo(lis,conslis)

	print(P)

@doit
def test_kk2():
	lis = [0,1,2,3,4,5]
	conslis = [[0,1],[0,2],[0,3],[0,4],[1,2],[1,4],[1,5],[2,3],[2,5],[3,4],[3,5],[4,5]]

	P = kk_algo(lis,conslis)

	print(P)



def type_check(sym_object):
	print(type(sym_object))
	
'''
@doit
def __test__():
	lnegth_j = sp.sqrt(sp.Sum((Pi[j,d]-p[d])**2, d_range)).doit()
	potential_j = Ki[j] * (lnegth_j - Li[j]) ** 2 /2
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
	type_check(lnegth_j)
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
$${lnegth_j}$$
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
