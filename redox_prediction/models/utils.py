#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:00:31 2022

@author: ljia
"""
from tensorflow.keras import layers


def split_data(inputs, *indices):
	outputs = []
	for item in inputs:
		for idx in indices:
			outputs.append([item[i] for i in idx])
	return tuple(outputs)


class Identity(layers.Layer):
    """A placeholder identity operator that is argument-insensitive.
    """
    def __init__(self):
        super(Identity, self).__init__()


    def call(self, x):
        """Return input."""
        return x