# Copyright (c) 2018/4/19 Hu Zhiming JimmyHu@pku.edu.cn All Rights Reserved.

# InceptionV3 class is used to calculate the bottleneck values(output values of InceptionV3) of an input image.


#################### Libs. ####################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image 

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


class InceptionV3(object):
	def __init__(self):
		
		# member variables.
		# the URL of the pre-trained model.
		self.HUB_MODULE = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
		# the model spec.
		self.Module_Spec = hub.load_module_spec(self.HUB_MODULE)
		# the image size that is required by this model.
		self.Module_Height, self.Module_Width = hub.get_expected_image_size(self.Module_Spec)
		self.Modelu_Depth = hub.get_num_image_channels(self.Module_Spec)
		# A module is understood as instrumented for quantization with TF-Lite
		# if it contains any of these ops.
		self.FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars', 'FakeQuantWithMinMaxVarsPerChannel')
				  
		# the size of our input images.
		self.ImageHeight = self.Module_Height
		self.ImageWidth = self.Module_Width
		self.ImageChannels = self.Modelu_Depth
		
		# Set up the pre-trained graph.
		self.graph, self.bottleneck_tensor, self.resized_input_tensor, self.wants_quantization = self.create_module_graph(self.Module_Spec)

	def create_module_graph(self, module_spec):
		"""
		Creates a graph and loads Hub Module into it.

		Args:
			module_spec: the hub.ModuleSpec for the image module being used.

		Returns:
			graph: the tf.Graph that was created.
			bottleneck_tensor: the bottleneck values output by the module.
			resized_input_tensor: the input images, resized as expected by the module.
			wants_quantization: a boolean, whether the module has been instrumented with fake quantization ops.
		"""
	
		with tf.Graph().as_default() as graph:
			resized_input_tensor = tf.placeholder(tf.float32, [None, self.ImageHeight, self.ImageWidth, self.ImageChannels])
			m = hub.Module(module_spec)
			bottleneck_tensor = m(resized_input_tensor)
			wants_quantization = any(node.op in self.FAKE_QUANT_OPS for node in graph.as_graph_def().node)
		return graph, bottleneck_tensor, resized_input_tensor, wants_quantization
	
	
	def CalculateBottlenecks(self, imagepath):
		'''Calculate the Bottleneck values of the given image.'''
		img = Image.open(imagepath)  
		# resize the image.
		img = img.resize((self.ImageWidth, self.ImageHeight))  
		# print(type(img))
		# PIL image -> ndarray.
		img = np.array(img)
		# add a new axis.
		img = img[np.newaxis, :]
		img = img.astype('float32')
		img = img* (1. / 255)
		# print(img.shape)
		# print(img.dtype)
		
		# print(type(img))
		# print(img.shape)
		with tf.Session(graph = self.graph) as sess:
			# Initialize all weights: for the module to their pretrained values,
			# and for the newly added retraining layer to random initial values.s
			init = tf.global_variables_initializer()
			sess.run(init)
			# Run Hub Module on the input data and calculate the bottleneck values.
			bottleneck_values = sess.run(self.bottleneck_tensor, {self.resized_input_tensor: img})
			# (1, 2048) --> (2048, )
			bottleneck_values = np.squeeze(bottleneck_values)
		return bottleneck_values	
		

if __name__ == '__main__':
	inceptionV3 = InceptionV3()
	print('Required Input Image Size: ')
	print('Width: {}'.format(inceptionV3.ImageWidth))
	print('Height: {}'.format(inceptionV3.ImageHeight))
	print('Channels: {}'.format(inceptionV3.ImageChannels))
	inceptionV3.CalculateBottlenecks('TestImages/1/1.jpg')