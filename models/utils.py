#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:00:31 2022

@author: ljia
"""

def split_data(inputs, train_index, test_index):
	outputs = []
	for item in inputs:
		outputs.append([item[i] for i in train_index])
		outputs.append([item[i] for i in test_index])
	return tuple(outputs)