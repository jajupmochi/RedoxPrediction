#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:52:49 2022

@author: ljia
"""

import pickle


def scores_to_latex(	fn_res):
	results = pickle.load(open(fn_res, 'rb'))
	perfs = results['perfs']
	ratio = 1
	for metrics, values in perfs.items():
		print(metrics)
		string = ''
		for set_, ls in values.items():
			string += set_ + ' ' + ' '.join(['& %.2f' % (v * ratio) for v in ls[:-1]]) + '$\pm$' + ('%.2f' % (ls[-1] * ratio)) + ' \\\\\n'
		print(string)


if __name__ == '__main__':
 	fn_res = '../outputs/miccio/poly200/100_trials/results.pkl'
 	# fn_res = '../outputs/miccio/poly200+sugarmono/results.pkl'
 	scores_to_latex(fn_res)
