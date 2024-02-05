#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:23:15 2022

@author: ljia
"""

# #%% Tensorflow
#
# import tensorflow as tf
# from tensorflow.keras import layers
#
#
# class GlobalAverageMaxPooling1D(layers.Layer):
#
# 	def __init__(self, **kwargs):
# 		super().__init__(**kwargs)
# # 		self.ave_pooling = layers.GlobalAveragePooling1D()
# # 		self.max_pooling = layers.GlobalMaxPooling1D()
#
#
# 	def call(self, inputs):
# 		# if inputs is a ragged tensor.
# 		if isinstance(inputs, tf.RaggedTensor):
# 			ave = tf.math.reduce_mean(inputs, axis=1)
# 			max_ = tf.math.reduce_max(inputs, axis=1)
# 			rst = tf.concat((ave, max_), axis=1)
# 			rst = tf.stack([i for i in rst])
# 			return rst
# 		elif isinstance(inputs, list):
# 			ave = tf.stack([tf.math.reduce_mean(i, axis=0) for i in inputs])
# 			max_ = tf.stack([tf.math.reduce_max(i, axis=0) for i in inputs])
# 			rst = tf.concat((ave, max_), axis=1)
# 			return rst
# 		else:
# 			ave = tf.math.reduce_mean(inputs, axis=1)
# 			max_ = tf.math.reduce_max(inputs, axis=1)
# 			rst = tf.concat((ave, max_), axis=1)
# 			return rst
#
#
# class GlobalAveragePooling1D(layers.Layer):
#
# 	def __init__(self, **kwargs):
# 		super().__init__(**kwargs)
#
#
# 	def call(self, inputs):
# 		# if inputs is a ragged tensor.
# 		if isinstance(inputs, tf.RaggedTensor):
# 			rst = tf.math.reduce_mean(inputs, axis=1)
# 			rst = tf.stack([i for i in rst])
# 			return rst
# 		elif isinstance(inputs, list):
# 			rst = tf.stack([tf.math.reduce_mean(i, axis=0) for i in inputs])
# 			return rst
# 		else:
# 			rst = tf.math.reduce_mean(inputs, axis=1)
# 			return rst
#
#
# class GlobalMaxPooling1D(layers.Layer):
#
# 	def __init__(self, **kwargs):
# 		super().__init__(**kwargs)
#
#
# 	def call(self, inputs):
# 		# if inputs is a ragged tensor.
# 		if isinstance(inputs, tf.RaggedTensor):
# 			rst = tf.math.reduce_max(inputs, axis=1)
# 			rst = tf.stack([i for i in rst])
# 			return rst
# 		elif isinstance(inputs, list):
# 			rst = tf.stack([tf.math.reduce_max(i, axis=0) for i in inputs])
# 			return rst
# 		else:
# 			rst = tf.math.reduce_max(inputs, axis=1)
# 			return rst


#%% Pytorch

import torch
from torch_scatter import scatter


def torch_sum(input_, dim=None, **kwargs):
	return torch.sum(input_, dim=dim)


def torch_mean(input_, node_mask=None, dim=None, **kwargs):
	sum_ = torch.sum(input_, dim=dim)
	count = torch.count_nonzero(node_mask, dim=dim)
	mean = torch.divide(sum_, count.view(-1, 1))
	return mean


def torch_max(input_, node_mask=None, dim=None, **kwargs):
	# Node mask is necessary as there might be all negative features.
	size = input_.shape[0]
	index = node_mask * (torch.arange(0, node_mask.shape[0]) + 1).view(-1, 1)
	index = index.view(-1)
	index_nz = index.nonzero().view(-1)
	src = input_.view(-1, input_.shape[-1])
	src = src[index_nz]
	index = index[index_nz].view(-1).long() - 1
	output = scatter(src, index, dim=-2, dim_size=size, reduce='max')
# 	output, _ = torch.max(input_, dim=dim)
	return output


def torch_meanmax(input_, node_mask=None, dim=None, **kwargs):
	mean = torch_mean(input_, node_mask=node_mask, dim=dim)
	max_ = torch_max(input_, node_mask=node_mask, dim=dim)
	output = torch.cat((mean, max_), dim=1)
	return output