import argparse
import sys
import sympy as sp
import typing


from symdoc import Markdown, doit, symfunc, gsym

cmdline_parser = argparse.ArgumentParser()
Markdown.add_parser_option(cmdline_parser)


M = Markdown('sqrt', title='Newton Rapthon for Python')
markdown, cmdline = M.markdown, M.cmdline

######################################################################
# 大域的な環境設定

import numpy as np
import sympy as sp

######################################################################

# 変数を定義しておく
global a,x
a,x = sp.var('a x')
F = sp.Function('F')

# NRを関数として定義しておく。
NR = x - F(x) / sp.diff(F(x),x)

# 任意の１変数関数と変数に関してNRに代入する関数
def makeNR(xa,FUNCTION_FORMULA):
	SUB_FORMULA = NR.subs(F(x),FUNCTION_FORMULA).doit()
	SUB_FORMULA_XA = SUB_FORMULA.subs(x,xa).doit()
	return SUB_FORMULA_XA

def calNRFormula(x1,f1):
	f1_sub = makeNR(x,f1)
	f1_cal = sp.lambdify(x,f1_sub)
	return f1_cal(x1)

def md_prog(f,a):
	x1 = calNRFormula(a,f)
	x2 = calNRFormula(x1,f)
	x3 = calNRFormula(x2,f)
	x4 = calNRFormula(x3,f)
	x5 = calNRFormula(x4,f)
	f_cal = sp.lambdify(x,f)
	y = f_cal(x5)
	markdown(
r'''
${f}=0$において
数列$x_n$を、$x_0 = {a}$として計算していくと
$$x_1 = {x1}$$
$$x_2 = {x2}$$
$$x_3 = {x3}$$
$$x_4 = {x4}$$
$$x_5 = {x5}$$
と収束していき、
${f} = 0$の近似解$α$は
$$α \simeq x_5 = {x5}$$
と求まります。
''',**locals())

@doit
def __first__():
	NRa= NR
	x_n, x_n1 = sp.var('x_n,x_{n+1}')
	f1 = x**2 - a
	f1_sub = makeNR(x_n,f1)
	x_x_n = 'x=x_n'
	f1_simple = sp.simplify(f1_sub)
	markdown(
r'''
#Newton Rapthon法を用いた平方根の計算

平方根を求めるプログラムを書いてみましょう。  
一般に平方根を求める計算にはNewton Rapthon法が用いられます。  
Newton Rapthon法とは、まず初めに、予想される真の解に近いと思われる値をひとつ選びます。  
次に、そこでグラフの接線を考え、そのx切片を計算します。  
このx切片の値は一般に、予想される真の解により近いものとなることが知られています。  
この値に対してそこでグラフの接線を考え、同じ操作を繰り返していく操作、これがNewton Rapthon法です。  
つまり、$x$の付近に$x_0$をとった次の漸化式を持つ数列$x_n$は$f(x)=0$の解に収束するのです。
$${x_n1} = \left.\left({NRa}\right) \right|_{{x=x_n}}$$
これを関数$f(x) = {f1}$に当てはめると、$a$の平方根が求まります。
すなわち数列
$${x_n1} = {f1_sub} = {f1_simple} = \frac{{a / x_n+ x_n}}{{2}}$$
はaの平方根に収束していくのです。
これにしたがって２,3,5の平方根を求めてみましょう。
''', **locals())
	fx2 = x**2 - 2
	fx3 = x**2 - 3
	fx5 = x**2 - 5

	md_prog(fx2,1)
	md_prog(fx3,2)
	md_prog(fx5,2)

	markdown(
r'''
# Newton Rapthon法の応用

他の関数にも応用してみます。次は２の三乗根、四乗根を求めてみましょう。
''', **locals())
	fx2_3 = x**3 - 2
	fx2_4 = x**4 - 2

	md_prog(fx2_3,1)
	md_prog(fx2_4,1)

	return
