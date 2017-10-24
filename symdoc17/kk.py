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

P = sp.IndexedBase('p')
L = sp.IndexedBase('L')
K = sp.IndexedBase('K')

# これはiベクトル
p = sp.IndexedBase('p')

j, d = [sp.Idx(*spec) for spec in [('j',n-1), ('d',dim)]]
j_range, d_range = [(idx, idx.lower, idx.upper) for idx in [j, d]]
lnegth_j = sp.sqrt(sp.Sum((P[j,d]-p[d])**2, d_range))
potential_j = K[j] * (lnegth_j - L[j]) ** 2 / 2
function_E = sp.Sum(potential_j,j_range)

@doit
def function():
	lnegth_j = sp.sqrt(sp.Sum((P[j,d]-p[d])**2, d_range))
	potential_j = K[j] * (lnegth_j - L[j]) ** 2 / 2
	function_E = sp.Sum(potential_j,j_range)
	grad = sp.diff(function_E,p[d])

	markdown(
r'''
$${lnegth_j}$$
$${potential_j}$$
$${function_E}$$
$${grad}$$
''', **locals())