#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:23:15 2022

@author: ljia
"""
import tensorflow as tf
from tensorflow.keras import layers


class GlobalAverageMaxPooling1D(layers.Layer):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.ave_pooling = layers.GlobalAveragePooling1D()
		self.max_pooling = layers.GlobalMaxPooling1D()


	def call(self, inputs):
		if isinstance(inputs, list):
			ave = tf.stack([tf.math.reduce_mean(i, axis=0) for i in inputs])
			max_ = tf.stack([tf.math.reduce_max(i, axis=0) for i in inputs])
			rst = tf.concat((ave, max_), axis=1)
			return rst
		else:
			ave = self.ave_pooling(inputs)
			max_ = self.max_pooling(inputs)
			rst = tf.concat((ave, max_), axis=1)
			return rst
