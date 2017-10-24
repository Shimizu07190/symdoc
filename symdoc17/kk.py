from functools import partial
import argparse
import sympy as sp
import numpy as np

from symdoc import Markdown, doit, symfunc, gsym

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
lnegth_j = sp.sqrt(sp.Sum((Pi[j,d]-p[d])**2, d_range))
potential_j = Ki[j] * (lnegth_j - Li[j]) ** 2 / 2
function_E = sp.Sum(potential_j,j_range)

@doit
def function():
	lnegth_j = sp.sqrt(sp.Sum((Pi[j,d]-p[d])**2, d_range).doit())
	potential_j = Ki[j] * sp.simplify(sp.expand((lnegth_j - Li[j]) ** 2)) / 2
	function_E = sp.Sum(potential_j,j_range)
	GRAD_0 = sp.diff(function_E,p[0])
	GRAD_1 = sp.diff(function_E,p[1])
	HESS_00 = sp.diff(grad_0,p[0])
	HESS_01 = sp.diff(grad_0,p[1])
	HESS_10 = sp.diff(grad_1,p[0])
	HESS_11 = sp.diff(grad_1,p[1])

	markdown(
r'''
$${lnegth_j}$$
$${potential_j}$$
$${function_E}$$
$${GRAD_0}$$
$${GRAD_1}$$
$${HESS_00}$$
$${HESS_01}$$
$${HESS_10}$$
$${HESS_11}$$
''', **locals())

def cal_loop():
	Carams = [Pi, Li, Ki, p, n]
	grad_0,grad_1 = [sp.lambdify((*Carams, j),f) for f in [GRAD_0, GRAD_1]]
	hess_00, hess_01, hess_11 = [sp.lambdify((*Carams, j),f) for f in [HESS_00, HESS_01, HESS_11]]